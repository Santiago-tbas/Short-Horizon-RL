
from statisticalrl_environments.MABs.StochasticBandits import MAB

import numpy as np

class MABofMDP: #TODO: to make a wrapper to MAB, we need reward distributions!
    def __init__(self, mdp, policies,max_step=np.infty):
        self.mdp_env = mdp
        self.policies = policies
        self.max_step = max_step

        #super(MABofMDP, self).__init__(self.mab.arms, distributionType=self.mab.distribution,
         #                                  structureType=self.mab.structure, structureParameter=self.mab.parameter,
         #                                  name=self.name)

    def step(self, policy):
        cumreward = 0.
        done = False
        observation, reward, info  = self.mdp_env.reset()
        policy.reset()
        t = 0
        while (not done) and (t<self.max_step):
            action = policy.action(observation)  # Get action
            newobservation, reward, done, truncated, info = self.mdp_env.step(action)
            policy.update(observation, action, reward, newobservation)  # Update policy
            observation = newobservation
            cumreward += reward
            t += 1
        return cumreward #(0,cumreward,False,False,{})





class QuantizedMAB(MAB):
    def __init__(self,mab,quantization_range,rewardmax):
        self.mab= mab
        self.quantization_range= quantization_range #integer range between 0 and 100 of tol.
        self.reward_max=rewardmax
        self.name="Q"+self.mab.name+"-tol"+str(quantization_range)
        super(QuantizedMAB, self).__init__(self.mab.arms, distributionType=self.mab.distribution, structureType=self.mab.structure, structureParameter=self.mab.parameter, name=self.name)


    def step(self, action):
        observation, reward, done, truncated, info = self.mab.step(action)
        obs_reward= np.floor(reward*100/self.reward_max/self.quantization_range)*self.quantization_range*self.reward_max/100
        info["quantization"]=self.quantization_range
        return obs_reward,reward,done,truncated,info

class BatchMAB(MAB):
    def __init__(self,mab,batchsize):
        self.mab = mab
        self.batchsize = batchsize
        self.name = "Batch"+self.mab.name+"-batch"+str(batchsize(0))
        self.round = 0
        super(BatchMAB, self).__init__(self.mab.arms, distributionType=self.mab.distribution, structureType=self.mab.structure, structureParameter=self.mab.parameter, name=self.name)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.round = 0
        info["nextbatchsize"]=self.batchsize(self.round)
        return observation, info

    def step(self, action):
        """action = [4,3,2]"""
        B= self.batchsize(self.round)
        assert len(action)==B
        batchreward = []
        batchobservation=[]
        batchmean=[]
        info={}
        for aa in action:
            observation, reward, done, truncated, info = self.mab.step(aa)
            batchobservation.append(observation)
            batchreward.append(reward)
            batchmean.append(info["mean"])
        self.round=self.round+1
        info["nextbatchsize"]=B
        info["mean"]=sum(batchmean)
        return (batchobservation,batchreward,False,False,info)




from statisticalrl_environments.MABs.StochasticBandits import BinomialBandit
class BatchQBinMAB(BatchMAB):
    def __init__(self,probabilities,batchsize,quantization_range,repetitions,name):
        # BinomialBandit
        # Base bandit: mab
        # reward distribution or each arm:  Sum of 200 Bernoulli, i.e. Binomial.
        #self.nameActions = action_names
        super(BatchQBinMAB, self).__init__(QuantizedMAB(BinomialBandit(means=probabilities,repetitions=repetitions),quantization_range,repetitions),batchsize)
        self.name = name