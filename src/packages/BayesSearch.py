from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# implemented using 
# https://towardsdatascience.com/bayesian-optimization-a-step-by-step-approach-a1cb678dd2ec
class BayesSearch:
    def __init__(
        self,
        data,
        target_function,
        sampling_function,
        n_iter,
        init_ex_count=20,
        gp_ex_count=1000,
    ):
        self.target_function = target_function
        self.gp_reg = GaussianProcessRegressor()
        self.output = pd.DataFrame(
            columns=["WindowSize", "SilenceThreshold", "Acc", "EI"]
        )  ## SB_COmment - any reason for EI over UCB or PI (Upper Confidence Bound/Probability of Improvement)
        self.sampling_function = sampling_function
        self.n_iter = n_iter
        self.data = data
        self.init_ex_count = init_ex_count
        self.gp_ex_count = gp_ex_count
        self.distances_ = []

        self.h = None
        self.y = None
        self._generate_initial()

    def _generate_initial(self):
        print(f"Initializing the {self.init_ex_count} hyper-parameters")

        self.h = self.sampling_function(self.init_ex_count)
        self.y = self.target_function(self.h, self.data)

    def expected_improvement(self, h_new):
        mean_y_new, sigma_y_new = self.gp_reg.predict(
            np.array([h_new]), return_std=True
        )
        sigma_y_new = sigma_y_new.reshape(-1, 1)
        if sigma_y_new == 0.0:
            return 0.0

        mean_y = self.gp_reg.predict(self.h)
        max_mean_y = np.max(mean_y)
        z = (mean_y_new - max_mean_y) / sigma_y_new
        exp_imp = (mean_y_new - max_mean_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)

        return exp_imp

    def next_params(self, explore_exploit_ratio=0.2):
        min_ei = np.inf
        max_ei = 0
        h_optimal = None
        h_new_sample = self.sampling_function(self.gp_ex_count)

        for x_new in h_new_sample:
            # response = minimize(fun=self.expected_improvement, x0=x_new, method='L-BFGS-B')
            # if response.fun < min_ei:
            #    min_ei = response.fun
            #    h_optimal = response.x
            exp_imp = self.expected_improvement(x_new)
            if exp_imp < min_ei:
                min_ei = exp_imp
                h_optimal = x_new
            if exp_imp > max_ei:
                max_ei = exp_imp
                h_optimal = x_new

        print("Optimal H: ", h_optimal)

        if np.random.rand() < explore_exploit_ratio:
            return h_optimal, max_ei
        else:
            return h_optimal, min_ei

    def optimize(self):
        y_max_ind = np.argmax(self.y)
        y_max = self.y[y_max_ind]
        optimal_h = self.h[y_max_ind]
        optimal_ei = None

        for i in range(self.n_iter):
            self.gp_reg.fit(self.h, self.y)
            h_next, ei = self.next_params()
            y_next = self.target_function(np.array([h_next]), self.data)
            print("acc: ", y_next)

            self.h = np.concatenate((self.h, np.array([h_next])))
            self.y = np.concatenate((self.y, np.array(y_next)))

            if y_next[0] > y_max:
                y_max = y_next[0]
                optimal_h = h_next
                optimal_ei = ei

            if i == 0:
                prev_h = h_next
            else:
                self.distances_.append(np.linalg.norm(prev_h - h_next))
                prev_h = h_next

            # self.best_samples_ = self.best_samples_.append({"y": y_max, "ei": optimal_ei},ignore_index=True)

        return optimal_h, y_max
