from unityagents import UnityEnvironment
import numpy as np


import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

env = UnityEnvironment(file_name="Reacher_Windows_x86_64\Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)

# size of each action
action_size = brain.vector_action_space_size
print("Size of each action:", action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    "There are {} agents. Each observes a state with length: {}".format(
        states.shape[0], state_size
    )
)
print("The state for the first agent looks like:", states[0])

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)


def ddpg(n_episodes=300, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    printed = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]

            # next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done,t)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_deque)
            ),
            end="",
        )
        torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
        torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        # if mean(scores_deque)>30 and not printed:
        #     print('env solved')
        #     printed = True

    return scores


scores = ddpg()
print(scores)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()