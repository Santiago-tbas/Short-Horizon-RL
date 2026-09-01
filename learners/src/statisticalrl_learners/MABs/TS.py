
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

'''
Bernoulli distributions — Beta-Bernoulli conjugate prior-posterior.

For Gaussian bandits (known variance), use GaussianTS from
statisticalrl_learners.MABs.GaussianTS which implements the exact
Normal-Normal conjugate Bayesian update.
'''
class TS(BanditAgent):
    """Thompson Sampling for Bernoulli bandits (Beta-Bernoulli conjugate)."""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        BanditAgent.__init__(self, self.nbArms, name="TS")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.zeros(self.nbArms)

    def play(self):
        return randmax(self.theta)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

        self.theta = [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1) for a in range(self.nbArms)]