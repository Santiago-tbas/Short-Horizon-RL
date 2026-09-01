import numpy as np
from statisticalrl_learners.MABs.TS import TS
from statisticalrl_learners.Generic.utils import randmax

class Algo1(TS):
    """
    Reference: Merlis & Mannor, "Lenient Regret for Multi-Armed Bandits" (AAAI 2021).
    """
    def __init__(self, nbArms, epsilon=0.1):
        super().__init__(nbArms)
        self.epsilon = epsilon
        self.agentname = f"e-TS (eps={epsilon})"
        self.reset()

    def reset(self):
        super().reset()
        self.empMeans = np.zeros(self.nbArms)
        self._sample_theta()


    def _sample_theta(self):
        theta = np.zeros(self.nbArms)
        for a in range(self.nbArms):
            mu_hat = self.empMeans[a]
            if mu_hat > 1.0 - self.epsilon:
                # Deterministic exploitation
                theta[a] = mu_hat
            else:
                # Attenuated exploration using modified Beta-Binomial parameters
                S_a = self.cumRewards[a]
                N_a = self.nbDraws[a]
                alpha = S_a / (1.0 - self.epsilon) + 1.0
                beta = N_a + 2.0 - alpha
                
                # Safeguards: ensure alpha and beta are strictly positive for np.random.beta
                alpha = max(1e-9, alpha)
                beta = max(1e-9, beta)
                
                Y = np.random.beta(alpha, beta)
                theta[a] = (1.0 - self.epsilon) * Y
        self.theta = theta

    def play(self):
        return randmax(self.theta)

    def update(self, arm, reward):
        
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.empMeans[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self._sample_theta()
