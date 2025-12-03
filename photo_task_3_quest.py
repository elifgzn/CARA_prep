class QuestPlusStaircase:
    """
    QUEST+ implementation with the same settings as the main CDT experiment.
    """

    def __init__(self, target_type):
        self.s_grid = np.linspace(logit(0.05), logit(0.90), 61)
        self.alpha_grid = np.linspace(logit(0.05), logit(0.90), 61)
        self.beta_grid = np.geomspace(1.0, 12.0, 25)
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])
        self.gamma = 0.5
        self.target_type = target_type

        if target_type == "high":
            alpha_mu = logit(0.48)
        elif target_type == "low":
            alpha_mu = logit(0.33)
        else:
            alpha_mu = logit(0.40)
        alpha_sd = 1.0

        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
        self.prior_alpha /= self.prior_alpha.sum()

        beta_mean = 2.5
        beta_gsd = 2.0
        ln_beta_mean = np.log(beta_mean)
        ln_beta_sd = np.log(beta_gsd)

        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - ln_beta_mean) / ln_beta_sd) ** 2)
        self.prior_beta /= self.prior_beta.sum()

        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

        self.post_alpha = self.prior_alpha.copy()
        self.post_beta = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        self.trial_count = 0
        self.responses = []

    def psychometric(self, s_logit, alpha, beta, lapse):
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid

    def compute_entropy(self, posterior):
        posterior = posterior + 1e-12
        return -np.sum(posterior * np.log(posterior))

    def select_stimulus_entropy_fast(self):
        s_grid_subset = self.s_grid[::3]

        current_entropy = self.compute_entropy(self.post_alpha)
        best_stimulus = None
        max_info_gain = -np.inf

        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        beta_mean = np.sum(self.beta_grid * self.post_beta)
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)

        for s_logit in s_grid_subset:
            p_correct = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue

            post_alpha_correct = np.zeros_like(self.post_alpha)
            post_alpha_incorrect = np.zeros_like(self.post_alpha)

            for i, alpha in enumerate(self.alpha_grid):
                like_correct = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                like_incorrect = 1.0 - like_correct
                post_alpha_correct[i] = self.post_alpha[i] * like_correct
                post_alpha_incorrect[i] = self.post_alpha[i] * like_incorrect

            post_alpha_correct /= (post_alpha_correct.sum() + 1e-12)
            post_alpha_incorrect /= (post_alpha_incorrect.sum() + 1e-12)

            entropy_correct = self.compute_entropy(post_alpha_correct)
            entropy_incorrect = self.compute_entropy(post_alpha_incorrect)
            expected_entropy = p_correct * entropy_correct + p_incorrect * entropy_incorrect

            info_gain = current_entropy - expected_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit

        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid) // 2]
        return clamp_prop(inv_logit(best_stimulus))

    def select_stimulus_entropy(self):
        return self.select_stimulus_entropy_fast()

    def update(self, stimulus_prop, correct):
        s_logit = logit(clamp_prop(stimulus_prop))
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))

        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    prior_weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    likelihood = (
                        self.psychometric(s_logit, alpha, beta, lapse)
                        if correct
                        else (1.0 - self.psychometric(s_logit, alpha, beta, lapse))
                    )
                    new_post[i, j, k] = prior_weight * likelihood

        new_post /= (new_post.sum() + 1e-12)
        self.post_alpha = new_post.sum(axis=(1, 2))
        self.post_beta = new_post.sum(axis=(0, 2))
        self.post_lambda = new_post.sum(axis=(0, 1))

        self.trial_count += 1
        self.responses.append((s_logit, correct))

    def get_threshold_sd(self):
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        alpha_var = np.sum(self.post_alpha * (self.alpha_grid - alpha_mean) ** 2)
        return float(np.sqrt(alpha_var))

    def threshold_for_target(self, p_target):
        lambda_hat = np.sum(self.lambda_grid * self.post_lambda)
        max_achievable = 1.0 - lambda_hat
        if p_target > max_achievable:
            p_target = min(0.85, max_achievable - 0.02)

        best_diff = float("inf")
        best_s = 0.5
        for s_logit in self.s_grid:
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += weight * self.psychometric(s_logit, alpha, beta, lapse)

            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s = inv_logit(s_logit)
        return clamp_prop(best_s)
