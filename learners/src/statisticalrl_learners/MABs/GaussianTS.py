from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

'''
Gaussian distributions — Normal-Normal conjugate prior-posterior.
'''
class GaussianTS(BanditAgent):
    def __init__(self, nbArms, sigma=1.0, mu0=0.0, sigma0=10.0):
        self.nbArms = nbArms
        self.sigma = sigma
        self.mu0 = mu0
        self.sigma0 = sigma0
        BanditAgent.__init__(self, self.nbArms, name="GaussianTS")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.random.normal(self.mu0, self.sigma0, self.nbArms)

    def play(self):
        return randmax(self.theta)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm] + reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        
        self.theta = []
        for a in range(self.nbArms):
            n_a = self.nbDraws[a]
            S_a = self.cumRewards[a]
            post_var = 1.0 / (1.0 / self.sigma0**2 + n_a / self.sigma**2)
            post_mean = post_var * (self.mu0 / self.sigma0**2 + S_a / self.sigma**2)
            self.theta.append(np.random.normal(post_mean, np.sqrt(post_var)))
