class QuestPlusStaircase:
    """
    QUEST+ Adaptive Staircase Implementation
    
    QUEST+ is a Bayesian adaptive psychophysical method that efficiently estimates
    psychometric function parameters by selecting stimuli that maximize information gain.
    
    Reference: Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive 
    psychometric method. Journal of Vision, 17(3):10, 1-27.
    
    The algorithm maintains probability distributions over psychometric function parameters
    and updates them using Bayes' rule after each trial. It selects the next stimulus
    by maximizing expected information gain (minimizing expected entropy).
    """

    def __init__(self, target_type):
        """
        Initialize QUEST+ staircase with parameter grids and prior distributions.
        
        Args:
            target_type: "high", "low", or other - determines the prior for alpha (threshold)
        
        Parameter Grids (in logit space for numerical stability):
        - s_grid: Stimulus levels (control proportions) to test
        - alpha_grid: Threshold parameter (50% point on psychometric curve)
        - beta_grid: Slope parameter (steepness of psychometric curve)
        - lambda_grid: Lapse rate (probability of random errors at easy levels)
        - gamma: Guess rate (fixed at 0.5 for 2AFC task)
        """
        
        # ====================================================================
        # PARAMETER GRIDS
        # ====================================================================
        # Stimulus grid: 61 levels from 5% to 90% control (in logit space)
        self.s_grid = np.linspace(logit(0.05), logit(0.90), 61)
        
        # Alpha (threshold): 61 levels from 5% to 90% (in logit space)
        # This is the control level where performance = 75% (midpoint between 50% guess and 100%)
        self.alpha_grid = np.linspace(logit(0.05), logit(0.90), 61)
        
        # Beta (slope): 25 levels from 1.0 to 12.0 (geometrically spaced)
        # Higher beta = steeper psychometric curve (more sensitive discrimination)
        self.beta_grid = np.geomspace(1.0, 12.0, 25)
        
        # Lambda (lapse rate): 5 levels from 0% to 6%
        # Accounts for random errors even at easy stimulus levels
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])
        
        # Gamma (guess rate): Fixed at 0.5 for 2-alternative forced choice
        self.gamma = 0.5
        
        self.target_type = target_type

        # ====================================================================
        # PRIOR DISTRIBUTIONS
        # ====================================================================
        # Prior for alpha (threshold) - Gaussian in logit space
        # Different target types have different expected thresholds
        # -------??????How are these determined?
        if target_type == "high":
            alpha_mu = logit(0.48)  # Expect higher threshold for "high" target
        elif target_type == "low":
            alpha_mu = logit(0.33)  # Expect lower threshold for "low" target
        else:
            alpha_mu = logit(0.40)  # Default middle threshold
        alpha_sd = 1.0  # Standard deviation in logit space

        # Gaussian prior over alpha grid
        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
        self.prior_alpha /= self.prior_alpha.sum()  # Normalize to sum to 1

        # Prior for beta (slope) - Log-normal distribution
        beta_mean = 2.5  # Expected slope value
        beta_gsd = 2.0   # Geometric standard deviation
        ln_beta_mean = np.log(beta_mean)
        ln_beta_sd = np.log(beta_gsd)

        # Log-normal prior over beta grid
        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - ln_beta_mean) / ln_beta_sd) ** 2)
        self.prior_beta /= self.prior_beta.sum()  # Normalize

        # Prior for lambda (lapse rate) - Uniform distribution
        # No strong prior belief about lapse rate
        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

        # ====================================================================
        # POSTERIOR DISTRIBUTIONS (initialized to priors)
        # ====================================================================
        # These will be updated after each trial using Bayes' rule
        self.post_alpha = self.prior_alpha.copy()
        self.post_beta = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        # Trial tracking
        self.trial_count = 0
        self.responses = []  # List of (stimulus, correct) tuples

    def psychometric(self, s_logit, alpha, beta, lapse):
        """
        Psychometric function: probability of correct response given stimulus and parameters.
        
        Formula: P(correct) = γ + (1 - γ - λ) * Φ(β(s - α))
        
        Where:
        - γ (gamma) = guess rate (0.5 for 2AFC)
        - λ (lapse/lambda) = lapse rate (random errors)
        - Φ = logistic function (sigmoid)
        - β (beta) = slope (steepness)
        - α (alpha) = threshold (50% point)
        - s = stimulus level
        
        Args:
            s_logit: Stimulus level in logit space
            alpha: Threshold parameter
            beta: Slope parameter
            lapse: Lapse rate
            
        Returns:
            Probability of correct response (0 to 1)
        """
        # Logistic sigmoid function: 1 / (1 + exp(-β(s - α)))
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        
        # Scale sigmoid to account for guessing and lapses
        # Range: [gamma, 1 - lapse] instead of [0, 1]
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid

    def compute_entropy(self, posterior):
        """
        Compute Shannon entropy of a probability distribution.
        this is a measure of uncertainty or randomness in a probability distribution
        
        Entropy H(p) = -Σ p(x) * log(p(x))
        
        Higher entropy = more uncertainty
        Lower entropy = more certainty (peaked distribution)
        
        Args:
            posterior: Probability distribution (sums to 1)
            
        Returns:
            Entropy value (non-negative)
        """
        # Add small constant to avoid log(0)
        posterior = posterior + 1e-12
        return -np.sum(posterior * np.log(posterior))

    def select_stimulus_entropy_fast(self):
        """
        Select next stimulus to maximize expected information gain.
        
        QUEST+ Strategy:
        1. For each candidate stimulus, simulate both possible outcomes (correct/incorrect)
        2. Calculate what the posterior distribution would be after each outcome
        3. Compute expected entropy across outcomes (weighted by their probabilities)
        4. Select stimulus that minimizes expected entropy (= maximizes info gain & minimizes uncertainty)
        
        This is a "fast" version that:
        - Uses only every 3rd stimulus level (reduces computation by ~3x)
        - Only updates alpha posterior (assumes beta and lambda are at their means)
        
        Returns:
            Selected stimulus level (control proportion, 0 to 1)
        """
        # Use subset of stimulus grid for speed (every 3rd point)
        s_grid_subset = self.s_grid[::3]

        # Current uncertainty about alpha (threshold)
        current_entropy = self.compute_entropy(self.post_alpha)
        
        best_stimulus = None
        max_info_gain = -np.inf

        # Use posterior means for beta and lambda (simplification)
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        beta_mean = np.sum(self.beta_grid * self.post_beta)
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)

        # Evaluate each candidate stimulus
        for s_logit in s_grid_subset:
            # Predict probability of correct response at this stimulus level
            p_correct = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            
            # Skip if outcome is too certain (no information gain)
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue

            # Simulate posterior distributions for both possible outcomes
            post_alpha_correct = np.zeros_like(self.post_alpha)
            post_alpha_incorrect = np.zeros_like(self.post_alpha)

            # For each possible alpha value, compute likelihood of each outcome
            for i, alpha in enumerate(self.alpha_grid):
                like_correct = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                like_incorrect = 1.0 - like_correct
                
                # Bayes' rule: posterior ∝ prior × likelihood
                post_alpha_correct[i] = self.post_alpha[i] * like_correct
                post_alpha_incorrect[i] = self.post_alpha[i] * like_incorrect

            # Normalize posteriors
            post_alpha_correct /= (post_alpha_correct.sum() + 1e-12)
            post_alpha_incorrect /= (post_alpha_incorrect.sum() + 1e-12)

            # Compute entropy for each possible outcome
            entropy_correct = self.compute_entropy(post_alpha_correct)
            entropy_incorrect = self.compute_entropy(post_alpha_incorrect)
            
            # Expected entropy = weighted average across outcomes
            expected_entropy = p_correct * entropy_correct + p_incorrect * entropy_incorrect

            # Information gain = reduction in entropy
            info_gain = current_entropy - expected_entropy
            
            # Track stimulus with maximum information gain
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit

        # Fallback: if no good stimulus found, use middle of range
        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid) // 2]
        
        # Convert from logit space back to proportion (0 to 1)
        return clamp_prop(inv_logit(best_stimulus))

    def select_stimulus_entropy(self):
        """
        Wrapper for stimulus selection method.
        Currently uses the fast entropy-based method.
        """
        return self.select_stimulus_entropy_fast()

    def update(self, stimulus_prop, correct):
        """
        Update posterior distributions after observing a trial outcome.
        
        Uses Bayes' rule to update beliefs about psychometric function parameters:
        P(θ|data) ∝ P(data|θ) × P(θ)
        
        Where:
        - θ = (alpha, beta, lambda) are the parameters
        - data = (stimulus, response)
        - P(θ) = current posterior (prior for this update)
        - P(data|θ) = likelihood from psychometric function
        - P(θ|data) = new posterior
        
        Args:
            stimulus_prop: Stimulus level presented (control proportion, 0-1)
            correct: Whether response was correct (True/False)
        """
        # Convert stimulus to logit space
        s_logit = logit(clamp_prop(stimulus_prop))
        
        # Create 3D joint posterior over all parameters
        # Dimensions: [alpha, beta, lambda]
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))

        # Compute posterior for each parameter combination
        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    # Prior: current posterior (assuming independence)
                    prior_weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    
                    # Likelihood: P(response | stimulus, parameters)
                    if correct:
                        likelihood = self.psychometric(s_logit, alpha, beta, lapse)
                    else:
                        likelihood = 1.0 - self.psychometric(s_logit, alpha, beta, lapse)
                    
                    # Bayes' rule: posterior ∝ prior × likelihood
                    new_post[i, j, k] = prior_weight * likelihood

        # Normalize joint posterior to sum to 1
        new_post /= (new_post.sum() + 1e-12)
        
        # Marginalize to get individual parameter posteriors
        # (sum over other dimensions to get marginal distribution)
        self.post_alpha = new_post.sum(axis=(1, 2))    # Sum over beta and lambda
        self.post_beta = new_post.sum(axis=(0, 2))     # Sum over alpha and lambda
        self.post_lambda = new_post.sum(axis=(0, 1))   # Sum over alpha and beta

        # Record trial
        self.trial_count += 1
        self.responses.append((s_logit, correct))

    def get_threshold_sd(self):
        """
        Get standard deviation of threshold (alpha) posterior distribution.
        
        This quantifies uncertainty about the threshold estimate.
        Lower SD = more confident estimate (after more informative trials).
        
        Returns:
            Standard deviation of alpha posterior
        """
        # Compute mean of alpha posterior
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        
        # Compute variance: E[(X - μ)²]
        alpha_var = np.sum(self.post_alpha * (self.alpha_grid - alpha_mean) ** 2)
        
        # Return standard deviation
        return float(np.sqrt(alpha_var))

    def threshold_for_target(self, p_target):
        """
        Find stimulus level that yields a target performance level.
        
        This is used to estimate the threshold (e.g., 75% correct point) from
        the current posterior distribution over psychometric function parameters.
        
        Method:
        1. For each candidate stimulus level
        2. Compute expected performance by averaging psychometric function
           over all parameter combinations (weighted by posterior probabilities)
        3. Find stimulus level closest to target performance
        
        Args:
            p_target: Target performance level (e.g., 0.75 for 75% correct)
            
        Returns:
            Stimulus level (control proportion) that yields target performance
        """
        # Estimate lapse rate from posterior
        lambda_hat = np.sum(self.lambda_grid * self.post_lambda)
        
        # Maximum achievable performance is limited by lapse rate
        max_achievable = 1.0 - lambda_hat
        
        # Adjust target if it exceeds maximum achievable
        if p_target > max_achievable:
            p_target = min(0.85, max_achievable - 0.02)

        # Search for stimulus level closest to target performance
        best_diff = float("inf")
        best_s = 0.5  # Default to middle
        
        for s_logit in self.s_grid:
            # Compute expected performance at this stimulus level
            # by averaging over all parameter combinations
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        # Weight by posterior probability of this parameter combination
                        weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += weight * self.psychometric(s_logit, alpha, beta, lapse)

            # Track stimulus closest to target performance
            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s = inv_logit(s_logit)  # Convert back to proportion
        
        return clamp_prop(best_s)
