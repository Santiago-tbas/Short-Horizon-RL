from statisticalrl_environments.MABs.StochasticBandits import MAB


import numpy as np



class QuantizedMAB(MAB):
    def __init__(self,mab,quantization_range,rewardmax):
        self.mab= mab
        self.quantization_range= quantization_range #integer range between 0 and 100 of tol.
        self.reward_max=rewardmax
        self.name="Q"+self.mab.name+"-tol"+str(quantization_range)
        super(QuantizedMAB, self).__init__(self.mab.arms, distributionType=self.mab.distribution, structureType=self.mab.structure, structureParameter=self.mab.parameter, name=self.name)


    def step(self, action):
        observation, reward, done, truncated, info = self.mab.step(action)
        if self.quantization_range>0:
            obs_reward= np.floor(reward*100/self.reward_max/self.quantization_range)*self.quantization_range*self.reward_max/100
        else:
            obs_reward= reward
        info["quantization"]=self.quantization_range
        return obs_reward,reward,done,truncated,info

class BatchMAB(MAB):
    def __init__(self,mab,batchsize):
        self.mab = mab
        # Accept a plain list (picklable for multiprocessing) or a callable.
        if callable(batchsize):
            self.batchsize = batchsize
        else:
            _sizes = list(batchsize)
            self.batchsize = lambda ell: _sizes[ell] if ell < len(_sizes) else 1
        self.name = "Batch"+self.mab.name+"-batch"+str(self.batchsize(0))
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
        info["nextbatchsize"]=self.batchsize(self.round)
        info["mean"]=sum(batchmean)
        return (batchobservation,batchreward,False,False,info)