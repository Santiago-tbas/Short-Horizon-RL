import pickle
import time
import os
import inspect


def _safe_reset(learner, observation):
    sig = inspect.signature(learner.reset)
    if len(sig.parameters) > 0:
        learner.reset(observation)
    else:
        learner.reset()


def _safe_play(learner, state):
    sig = inspect.signature(learner.play)
    if len(sig.parameters) > 0:
        return learner.play(state)
    else:
        return learner.play()


def _safe_update(learner, state, action, reward, next_state):
    sig = inspect.signature(learner.update)
    if len(sig.parameters) >= 4:
        learner.update(state, action, reward, next_state)
    elif len(sig.parameters) == 2:
        learner.update(action, reward)
    else:
        try:
            learner.update(state, action, reward, next_state)
        except TypeError:
            learner.update(action, reward)


def dump(values, filename, tag, root_folder):
    filenameM = root_folder + filename + "_" + tag
    file = open(filenameM, 'wb')
    file.truncate(0)
    pickle.dump(values, file)
    file.close()
    return filenameM


def clear_auxiliaryfiles(env, root_folder):
    for file in os.listdir(root_folder):
        if file.startswith("cumMeans_" + env.name) or file.startswith("cumGaps_" + env.name):
            os.remove(root_folder + file)

def oneXpNoRender(env,learner,timeHorizon,root_folder):
    observation, info = env.reset()
    _safe_reset(learner, observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("[Info] New initialization of ", learner.name(), ' for environment ',env.name)
    #print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = _safe_play(learner, state)  # Get action
        observation, reward, done,truncated, info = env.step(action)
        _safe_update(learner, state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info["mean"]
        except TypeError:
            cummean += reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation, info=env.reset()
            #break

    #print("Cumreward: " + str(cumreward))
    #print("Cummean: " + str(cummean))
    return cummeans #cumrewards,cummeans


def oneXpNoRenderWithDump(env,learner,timeHorizon,root_folder):
    observation,info = env.reset()
    _safe_reset(learner, observation)
    cumreward = 0.
    cumrewards = []
    cummean = 0.
    cummeans = []
    print("[Info] New initialization of ", learner.name(), ' for environment ',env.name)
    #print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = _safe_play(learner, state)  # Get action
        observation, reward, done, truncated, info = env.step(action)
        _safe_update(learner, state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cumreward += reward
        try:
            cummean += info["mean"]
        except TypeError:
            cummean +=reward
        cumrewards.append(cumreward)
        cummeans.append(cummean)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation, info = env.reset() # converts an episodic MDP into an infinite time horizon MDP
            #break

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) +"_" + str(time.time())
    filename = dump(cummeans,"cumMeans",tag,root_folder)
    return filename

def oneRunOptWithDump(env, opti_learner, timeHorizon, root_folder):
 ## Cumlative reward of optimal policy:
    opttimeHorizon = min(max((1000000, timeHorizon)),10**8)
    cumReward_opti = oneXpNoRender(env, opti_learner, opttimeHorizon, root_folder=root_folder)
    gain =  cumReward_opti[-1] / len(cumReward_opti)
    #print("Average gain is ", gain)
 # TODO: remove one [] and update computeCumulativeRegrets accordingly.
    opti_cumgain = [[t * gain for t in range(timeHorizon)]]

    tag = env.name + "_" + opti_learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    filename= dump(opti_cumgain,"cumMeans",tag,root_folder)
    return filename


def oneXpBatchNoRenderWithDump(env, learner, timeHorizon, root_folder):
    observation, info = env.reset()
    learner.reset()
    B = info["nextbatchsize"]
    cumreward = 0
    cummean = 0
    cumgap = 0
    cumrewards = []
    cummeans = []
    cumgaps = []
    for t in range(timeHorizon):
        batchaction = learner.batchplay(B)  # Get action
        batchobservation, batchreward, done, truncated, info = env.step(batchaction) # Get response
        learner.batchupdate(batchaction, batchobservation)  # Update learners
        B = info["nextbatchsize"]
        #print({"batchobservation": batchobservation, "batchreward": batchreward, "pseudo-gap":max(env.means) * B-sum(batchreward), "info": info})
        cumreward += sum(batchreward)
        cummean += info["mean"]
        cumgap += max(env.means) * B-sum(batchreward)
        cumrewards.append(cumreward)
        cummeans.append(cummean)
        cumgaps.append(cumgap)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    #filenameR = dump(cumrewards, "cumRewards", tag, root_folder)
    filenameM = dump(cummeans, "cumMeans", tag, root_folder)
    filenameG = dump(cumgaps, "cumGaps", tag, root_folder)
    return filenameM,filenameG#filenameR, filenameM, filenameG

