import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from agents.agent import DDPG_Agent
from task import Task

def plot_scores(scores, rolling_window=100):
	"""Plot scores and optional rolling mean using specified window."""
	plt.plot(scores); plt.title("Scores");
	rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
	plt.plot(rolling_mean);
	plt.show()
	return rolling_mean

def run(agent, num_episodes, max_steps):
	scores = []
	max_avg_score = -np.inf

	for i_episode in range(1, num_episodes+1):
		state = agent.reset_episode() # receive initial observation state
		
		t = 0; total_reward = 0;
		while t < max_steps:
			# Select action from current policy
			action = agent.act(state) 
			# Execute action and get interaction 
			next_state, reward, done = task.step(action)
			total_reward += reward
			# Get one-step experience and update parameters
			agent.step(action, reward, next_state, done)
			t += 1
			
			if done:
				break
				
		scores.append(total_reward)
		
		if len(scores) >= 20:
			avg_score = np.mean(scores[-20:])
			if avg_score > max_avg_score:
					max_avg_score = avg_score
		if i_episode % 20 == 0:
				print('\rEpisode {}/{} | Max Average Score: {} | Current Average Score: {}'
					  .format(i_episode, num_episodes, max_avg_score, avg_score), end='')
		sys.stdout.flush()
		
	return scores

# Task setting (hover task)
init_pose = np.array([0., 0., .1, 0., 0., 0.])  #start: x=0, y= 0, z=2
target_pos = np.array([0., 0., 10.]) #goal: x=3, y= 3, z=10

task = Task(init_pose=init_pose, target_pos=target_pos)

# Hyper parameters 
num_episodes = 500 # max number of episodes to learn from
max_steps = 200	 # max steps in an episode
gamma = 0.95
tau = 0.001

# Network parameters
learning_rate_actor = 1e-5
learning_rate_critic = 1e-5

# Memory parametes
memory_size = 10000
batch_size = 64

# initialize agent instance
agent = DDPG_Agent(task, 
					learning_rate_actor=learning_rate_actor,
					learning_rate_critic=learning_rate_critic,
					gamma=gamma,
					tau=tau,
					buffer_size=memory_size,
					batch_size=batch_size) 

scores = run(agent, num_episodes, max_steps)
rolling_mean = plot_scores(scores, 50)