
from statisticalrl_environments.BatchMABs.wrappers import QuantizedMAB, BatchMAB
from statisticalrl_environments.MABs.Distributions import Binomial, Gaussian, TruncatedGaussian, Bernoulli

class BatchBandit(BatchMAB):
    def __init__(self, action_names, probabilities, batchsize, repetitions, name):
        self.nameActions = action_names
        super(BatchBandit, self).__init__(
            QuantizedMAB(BinomialBandit(means=probabilities, repetitions=repetitions), 0, repetitions),
            batchsize)
        self.name = name


class BatchGBandit(BatchMAB):
    def __init__(self, action_names, probabilities, batchsize, variance, name):
        self.nameActions = action_names
        super(BatchGBandit, self).__init__(
            QuantizedMAB(GaussianBandit(means=probabilities, variance=variance), 0, 1.),
            batchsize)
        self.name = name


class BatchTruncGBandit(BatchMAB):
    """Batch bandit with TruncatedGaussian arms on [low, high].

    The upper bound `high` is accessible as ``env.reward_max`` for use with
    KLinf_threshold.
    """
    def __init__(self, action_names, probabilities, batchsize, sigma=0.5,
                 low=-1.0, high=1.0, name="BatchTruncGBandit"):
        self.nameActions = action_names
        self.reward_max = high    # upper bound of support — pass to KLinf
        base = TruncatedGaussianBandit(means=probabilities, sigma=sigma, low=low, high=high)
        super(BatchTruncGBandit, self).__init__(
            QuantizedMAB(base, 0, high),
            batchsize)
        self.name = name


class BatchBernBandit(BatchMAB):
    """Batch bandit with Bernoulli arms on [0, 1].

    ``env.reward_max = 1.0`` for use with KLinf_threshold.
    """
    def __init__(self, action_names, probabilities, batchsize, name="BatchBernBandit"):
        self.nameActions = action_names
        self.reward_max = 1.0
        base = BernoulliBandit(means=probabilities)
        super(BatchBernBandit, self).__init__(
            QuantizedMAB(base, 0, 1.0),
            batchsize)
        self.name = name