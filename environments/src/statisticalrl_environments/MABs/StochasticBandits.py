import numpy as np
import statisticalrl_environments.MABs.Distributions as arms
import string


from gymnasium import Env, spaces
from gymnasium.utils import seeding


import statisticalrl_environments.MABs.rendering.textRenderer as tRendering

class MAB(Env):
    def __init__(self, arms, distributionType='unknown', structureType='unknown', structureParameter=None, name="MAB"):
        """given a list of arms, create the MAB environment"""
        self.name = name
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
        self.distribution = distributionType
        self.structure = structureType
        self.parameter = structureParameter


        self.rendermode = 'text'
        self.initializedRenderer = False
        self.renderers = {'':(), 'text': tRendering.textRenderer}
        self.nameActions = list(string.ascii_uppercase)[0:min(self.nbArms, 26)]

        self.states = range(0, 1)
        self.actions = range(0, self.nbArms)

        self.s= 0
        self.lastaction = None
        self.lastreward = 0


        self.action_space = spaces.Discrete(self.nbArms)
        self.observation_space = spaces.Discrete(1)


    #def generateReward(self, arm):
    #    return self.arms[arm].sample()


    def change_rendermode(self,rendermode):
        self.rendermode = rendermode
        self.initializedRenderer = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.np_random, seed = seeding.np_random(seed)
        self.lastaction = None
        return 0, {"mean": 0}

    def step(self, arm):
        """
        :param a: action
        :return:  (state, reward, IsDone?, IsTruncated?, meanreward)
        The meanreward is returned for information, it should not be given to the learner.
        """
        r = self.arms[arm].sample()
        self.lastaction = arm
        self.lastreward = r
        return 0, r, False, False, {"mean": self.means[arm]}


    def render(self, mode='human'):
        #     #Note that default mode is 'human' for open-ai-gymnasium
        if self.rendermode != '':
            if ((not self.initializedRenderer)):
                    try:
                        self.renderer = self.renderers[self.rendermode]()
                    except KeyError:
                        print("Invalid key '"+self.rendermode+"'. Please use one of the following keys: ", str([x for x in self.renderers.keys()]))
                    self.initializedRenderer = True
            self.renderer.render(self, self.s, self.lastaction,
                                     self.lastreward)






## some functions that create specific MABs

def BernoulliBandit(means, structure='unknown', parameter=None, name="MAB-Bernoulli"):
    """define a Bernoulli MAB from a vector of means"""
    return MAB([arms.Bernoulli(p) for p in means], distributionType='Bernoulli', structureType=structure,
               structureParameter=parameter, name=name)

def BinomialBandit(means, repetitions=200, name="MAB-Binomial"):
    """define a Bernoulli MAB from a vector of means"""
    s="-".join(str(m) for m in means)
    name = f'{name}{repetitions}-means-{s}'
    return MAB([arms.Binomial(repetitions,p) for p in means], distributionType='Binomial', structureType='unknown',
               structureParameter=None, name=name)


def GaussianBandit(means, vars, structure='unknown', parameter=None, name="MAB-Gaussian"):
    """define a Gaussian MAB from a vector of means"""
    return MAB([arms.Gaussian(m, v) for m,v in zip(means,vars)], distributionType='Gaussian', structureType=structure,structureParameter=parameter,name=name)




def TruncatedGaussianBandit(means, sigma=0.5, low=-1.0, high=1.0, name="MAB-TruncGaussian"):
    s = "-".join(str(m) for m in means)
    name = f'{name}-sigma{sigma}-means-{s}'
    return MAB([arms.TruncatedGaussian(p, sigma=sigma, low=low, high=high) for p in means],
               distributionType='TruncatedGaussian', structureType='unknown',
               structureParameter=None, name=name)


def RandomBernoulliBandit(Delta, K, name="MAB-RandomBernoulli"):
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand() * (1. - Delta)
    secondmaxMean = maxMean - Delta
    means = secondmaxMean * np.random.random(K)
    bestarm = np.random.randint(0, K)
    secondbestarm = np.random.randint(0, K)
    while (secondbestarm == bestarm):
        secondbestarm = np.random.randint(0, K)
    means[bestarm] = maxMean
    means[secondbestarm] = secondmaxMean
    return BernoulliBandit(means, name=name)
