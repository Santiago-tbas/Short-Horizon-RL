
from statisticalrl_environments.register import registerStatisticalRLenvironments,make
import numpy as np



class RandomAgent:
    def __init__(self,env):
        self.env=env

    def name(self):
        return "Random Agent"

    def reset(self,inistate):
        ()

    def play(self,state):
        if hasattr(self.env, 'batchsize'):
            B = self.env.batchsize(self.env.round)
            return [self.env.action_space.sample() for _ in range(B)]
        return self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        ()



def animate(env, learner, timeHorizon):
    print("Render mode:", str(env.rendermode))
    print("New initialization of ", learner.name())
    observation, info = env.reset()
    print("Initial state:" + str(observation))
    #env.render()
    learner.reset(observation)
    cumreward = 0.
    cumrewards = []
    for t in range(timeHorizon):
        state = observation
        env.render()
        action = learner.play(state)  # Get action
        observation, reward, done, truncated,  info = env.step(action)
        # print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
        learner.update(state, action, reward, observation)  # Update learners
        cumreward += sum(reward) if hasattr(env, 'batchsize') else reward
        cumrewards.append(cumreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break







def demo(envname):
     env = make(envname)
     print("-"*30+"\n"+env.name+"\n"+"-"*30)
     print("Available renderers: ",list(env.renderers.keys()))
     rendermode = np.random.choice(list(env.renderers.keys())[1:])
     if(envname=="random-100"):
         rendermode='text' #Force text rendering as others are farily slow.
     env.change_rendermode(rendermode)
     learner = RandomAgent(env)
     #learner = Human(env)
     #learner = UCRL3_lazy(env.observation_space.n, env.action_space.n, delta=0.05)
     animate(env, learner, 20)
     print("-"*30+"\n")

def print_registered_environments():
    print("-"*30+ "\nList of registered environments:\n"+ "-"*30)
    [print(k) for k in registerStatisticalRLenvironments.keys()]
    print("-"*30)

def random_environment():
    envname = np.random.choice(list(registerStatisticalRLenvironments.keys()))
    demo(envname)

def all_environments():
    for e in registerStatisticalRLenvironments:
        demo(e)

if __name__ == "__main__":

    print_registered_environments()
    random_environment()
    all_environments()



