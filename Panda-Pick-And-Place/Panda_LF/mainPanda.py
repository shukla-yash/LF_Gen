import os
import sys

import gym

import numpy as np
import gym_novel_gridworlds

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt



if __name__ == "__main__":

	env_id = 'NovelGridworld-v1'
	env = gym.make(env_id, map_width = 5, map_height = 5, map_length = 5, goal_env = 2)

	actionCnt = 7
	D = len(env.low)
	NUM_HIDDEN = 16
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 11

	agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	agent.set_explore_epsilon(MAX_EPSILON)
	action_space = ['W','A','D','U','C']
	
	t_step = 0
	episode = 0
	t_limit = 100
	reward_sum = 0
	reward_arr = []
	avg_reward = []

	env.reset()

	while True:
		#print env.toString()
		
		# get obseration from sensor
	

		obs = env.get_observation()
	
		# act 
		a = agent.process_step(obs,True)
		#print("Action at t="+str(t_step)+" is "+action_space[a])
		
		new_obs, reward, done, info = env.step(a)
		#print("Reward = "+str(reward))
		# give reward
		agent.give_reward(reward)
		reward_sum += reward
		
		t_step += 1
		
		if t_step > t_limit or done == True:
			
			# finish agent
			
			print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
			# print("epsilon is: ", MAX_EPSILON)
			reward_arr.append(reward_sum)
			avg_reward.append(np.mean(reward_arr[-50:]))
			# print("avg_reward is: ", reward_arr)			
			done = True
			t_step = 0
			agent.finish_episode()
		
			# update after every episode
			agent.update_parameters()
		
			# reset environment
			episode += 1
			# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
			# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'pogo_stick':0}, \
			# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'pogo_stick':0}, goal_env = 2)
			env.reset()
			reward_sum = 0
	
			# quit after some number of episodes
			if episode > 80000:
				print("reward arr is: ", len(reward_arr))
				plt.plot(avg_reward)
				# plt.savefig('avg_reward_jivko7.png')
				break


	# env_id = 'NovelGridworld-v0'
	# env = gym.make(env_id, map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'pogo_stick':0},
	# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'pogo_stick':0}, goal_env = 2 )

	# env.map_width = 10
	# env.map_height = 10
	# env.goal_env = 2

	# actionCnt = 5
	# D = len(env.low)
	# NUM_HIDDEN = 10
	# GAMMA = 0.95
	# LEARNING_RATE = 1e-3
	# DECAY_RATE = 0.99
	# MAX_EPSILON = 0.1
	# random_seed = 11
	
	# t_step = 0
	# episode = 0
	# t_limit = 100
	# reward_sum = 0
	# reward_arr = []
	# avg_reward = []

	# env.reset()

	# while True:
	# 	#print env.toString()
		
	# 	# get obseration from sensor
	

	# 	obs = env.get_observation()
	
	# 	# act 
	# 	a = agent.process_step(obs,True)
	# 	#print("Action at t="+str(t_step)+" is "+action_space[a])
		
	# 	new_obs, reward, done, info = env.step(a)
	# 	#print("Reward = "+str(reward))
	# 	# give reward
	# 	agent.give_reward(reward)
	# 	reward_sum += reward
		
	# 	t_step += 1
		
	# 	if t_step > t_limit or done == True:
			
	# 		# finish agent
			
	# 		print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
	# 		# print("epsilon is: ", MAX_EPSILON)
	# 		reward_arr.append(reward_sum)
	# 		avg_reward.append(np.mean(reward_arr[-10:]))
	# 		# print("avg_reward is: ", reward_arr)			
	# 		done = True
	# 		t_step = 0
	# 		agent.finish_episode()
		
	# 		# update after every episode
	# 		agent.update_parameters()
		
	# 		# reset environment
	# 		episode += 1
	# 		# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
	# 		# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'pogo_stick':0}, \
	# 		# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'pogo_stick':0}, goal_env = 2)
	# 		env.reset()
	# 		reward_sum = 0
	
	# 		# quit after some number of episodes
	# 		if episode > 6000:
	# 			# print("avg reward is: ", avg_reward)
	# 			plt.plot(avg_reward)
	# 			plt.savefig('avg_reward_jivko5.png')
	# 			break


	# env.reset()
	# t_step = 0
	# episode = 0
	# while episode < 10:
	# 	#print env.toString()
		
	# 	# get obseration from sensor
	# 	env.render()
	
	# 	# env.render()
	# 	obs = env.get_observation()
	
	# 	# act 
	# 	a = agent.process_step(obs,True)
	# 	#print("Action at t="+str(t_step)+" is "+action_space[a])
		
	# 	new_obs, reward, done, info = env.step(a)
	# 	# env.render()
	# 	#print("Reward = "+str(reward))
	# 	# give reward
	# 	agent.give_reward(reward)
	# 	reward_sum += reward
		
	# 	t_step += 1
		
	# 	if t_step > t_limit or done == True:
	# 		print("it is done")
	# 		# finish agent
			
	# 		print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
	# 		# print("epsilon is: ", MAX_EPSILON)
	# 		reward_arr.append(reward_sum)
	# 		avg_reward.append(np.mean(reward_arr[-10:]))
	# 		# print("avg_reward is: ", reward_arr)			
	# 		done = True
	# 		t_step = 0

	# 		# reset environment
	# 		episode += 1
	# 		# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
	# 		# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'pogo_stick':0}, \
	# 		# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'pogo_stick':0}, goal_env = 2)
	# 		env.reset()
	# 		# env.render()
	# 		reward_sum = 0