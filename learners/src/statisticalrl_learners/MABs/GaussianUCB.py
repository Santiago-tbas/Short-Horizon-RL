from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

"""UCB for Gaussian bandits with known variance sigma^2."""
class GaussianUCB(BanditAgent):
    def __init__(self, nbArms, sigma=1.0, delta=lambda t: 1.0 / t):
        self.nbArms = nbArms
        self.sigma = sigma
        self.delta = delta
        BanditAgent.__init__(self, self.nbArms, name="GaussianUCB")

    def reset(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.full(self.nbArms, np.inf)

    def play(self):
        return randmax(self.indexes)

    def update(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm] + reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        self.indexes = [self.means[a] + self.sigma * sqrt(2 * log(1/self.delta(self.time))/self.nbDraws[a]) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]
