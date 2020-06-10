import torch
from unityagents import UnityEnvironment
import numpy as np
from agent import Agent


def checkWeights(agent, env, brain_name, eps=0.05):
    file_weights = 'checkpoint_2.pth'
    agent.qnetwork_local.load_state_dict(torch.load(file_weights))

    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps)
        action = action.astype(np.int32)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0] 
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break 
    
    return score


env = UnityEnvironment(file_name="C:/Users/SalemAyadi/Documents/Salem/udacity_project/rl_navigator/Banana_Windows_x86_64/Banana.exe")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]


# number of actions and states
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

list_scores = []
for test in range(0,6):        
    score = checkWeights(agent, env, brain_name)
    list_scores.append(score)
    print ('current score', score)
avg_score =  np.mean(list_scores)
print('Average Score: ', avg_score)