import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent import Agent
from dqn import dqn
import os



env_path = os.path.join(os.getcwd(),"Banana_Windows_x86_64/Banana.exe")
env = UnityEnvironment(file_name = env_path)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]


# number of actions and states
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, seed=0, Dueling_Normal = 'Dueling')
scores = dqn(agent, env, brain_name, 'double')

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
