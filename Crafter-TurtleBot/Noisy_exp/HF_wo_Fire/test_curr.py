import os
import sys

import gym
import time
import numpy as np
# import gym_novel_gridworlds
import TurtleBot_v0

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt



def CheckTrainingDoneCallback(reward_array, done_array, env):

	done_cond = False
	reward_cond = False
	if len(done_array) > 40:
		if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
			if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
				done_cond = True

		if done_cond == True:
			if env < 3:
				if np.mean(reward_array[-40:]) > 730:
					reward_cond = True
			# else:
			# 	if np.mean(reward_array[-10:]) > 950:
			# 		reward_cond = True

		if done_cond == True and reward_cond == True:
			return 1
		else:
			return 0
	else:
		return 0





if __name__ == "__main__":

	no_of_environmets = 4

	width_array = [2.5,3.5,3.8,4]
	width_std = 1.5/6

	height_array = [2.5,3.5,3.8,4]
	height_std = 1.5/6
	no_trees_array = [1,1,3,4]
	trees_std = 4/6

	no_rocks_array = [0,1,2,2]
	rocks_std = 2/6

	crafting_table_array = [0,0,1,1]
	starting_trees_array = [0,0,0,0]
	starting_rocks_array = [0,0,0,0]
	type_of_env_array = [0,1,2,2]

	total_timesteps_array = []
	total_reward_array = []
	avg_reward_array = []
	final_timesteps_array = []
	final_reward_array = []
	final_avg_reward_array = []
	task_completion_array = []



	actionCnt = 5
	D = 83 #90 beams x 4 items lidar + 3 inventory items
	NUM_HIDDEN = 16
	GAMMA = 0.995
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 0

	# agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	# agent.set_explore_epsilon(MAX_EPSILON)
	action_space = ['W','A','D','U','C']
	total_episodes_arr = []

	for i in range(no_of_environmets):
		print("Environment: ", i)
		# i = 1

		width = width_array[i] + np.random.normal(-width_std, width_std)
		if width < 1.5:
			width = 1.5
		if width > 3:
			width = 3

		height = height_array[i] + + np.random.normal(-height_std, height_std)
		if height < 1.5:
			height = 1.5
		if height > 3:
			height = 3

		no_trees = round(no_trees_array[i] + np.random.normal(-trees_std,trees_std))
		if no_trees < 0:
			no_trees = 0

		no_rocks = round(no_rocks_array[i] + + np.random.normal(-trees_std,trees_std))
		if no_rocks < 0:
			no_rocks = 0

		crafting_table = crafting_table_array[i]
		starting_trees = starting_trees_array[i]
		starting_rocks = starting_rocks_array[i]
		type_of_env = type_of_env_array[i]

		if type_of_env == 0 or type_of_env == 1:
			if no_trees == 0 and no_rocks == 0:
				no_trees = 1

		if type_of_env == 2:
			if no_trees < 2 and no_rocks < 1:
				no_trees = 2
				no_rocks = 1

		print(f"Finally {width} width | {height} height | {no_trees} trees | {no_rocks} rocks | {type_of_env} goal ")

		final_status = False

		if i == 0:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
		else:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
			agent.load_model(0,0,i-1)
			agent.reset()
			print("loaded model")


		if i == no_of_environmets-1:
			final_status = True

		env_id = 'TurtleBot-v0'
		env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'stone_axe':0},
			initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'stone_axe':0}, goal_env = type_of_env, is_final = final_status)
		
		t_step = 0
		episode = 0
		t_limit = 600
		reward_sum = 0
		reward_arr = []
		avg_reward = []
		done_arr = []
		env_flag = 0

		env.reset()

		while True:
			
			# get obseration from sensor
			obs = env.get_observation()
		
			# act 
			a = agent.process_step(obs,True)
			
			new_obs, reward, done, info = env.step(a)

			# give reward
			agent.give_reward(reward)
			reward_sum += reward
			
			t_step += 1
			
			if t_step > t_limit or done == True:
				
				# finish agent
				if done == True:
					done_arr.append(1)
					task_completion_array.append(1)
				elif t_step > t_limit:
					done_arr.append(0)
					task_completion_array.append(0)
				
				print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")

				reward_arr.append(reward_sum)
				avg_reward.append(np.mean(reward_arr[-40:]))

				total_reward_array.append(reward_sum)
				avg_reward_array.append(np.mean(reward_arr[-40:]))
				total_timesteps_array.append(t_step)
		
				done = True
				t_step = 0
				agent.finish_episode()
			
				# update after every episode
				if episode % 10 == 0:
					agent.update_parameters()
			
				# reset environment
				episode += 1

				env.reset()
				reward_sum = 0

				env_flag = 0
				if i< 3:
					env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)
		
				# quit after some number of episodes
				if episode > 70000 or env_flag == 1:

					agent.save_model(0,0,i)
					total_episodes_arr.append(episode)

					break

	print("Total epsiode array is: ", total_episodes_arr)

	log_dir = 'logs_' + str(random_seed)
	os.makedirs(log_dir, exist_ok = True)

	total_timesteps_array = np.asarray(total_timesteps_array)
	print("size total_timesteps_array: ", total_timesteps_array.shape)
	
	total_reward_array = np.asarray(total_reward_array)
	print("size total_reward_array: ", total_reward_array.shape)

	avg_reward_array = np.asarray(avg_reward_array)
	print("size avg_reward_array: ", avg_reward_array.shape)

	total_episodes_arr = np.asarray(total_episodes_arr)
	print("size total_episodes_arr: ", total_episodes_arr.shape)

	task_completion_arr = np.asarray(task_completion_array)

	# final_timesteps_array = np.asarray(final_timesteps_array)
	# print("size final_timesteps_array: ", final_timesteps_array.shape)

	# final_reward_array = np.asarray(final_reward_array)
	# print("size final_reward_array: ", final_reward_array.shape)

	# final_avg_reward_array = np.asarray(final_avg_reward_array)
	# print("size final_avg_reward_array: ", final_avg_reward_array.shape)

	experiment_file_name_total_timesteps = 'randomseed_' + str(random_seed) + '_total_timesteps'
	path_to_save_total_timesteps = log_dir + os.sep + experiment_file_name_total_timesteps + '.npz'

	experiment_file_name_total_reward = 'randomseed_' + str(random_seed) + '_total_reward'
	path_to_save_total_reward = log_dir + os.sep + experiment_file_name_total_reward + '.npz'

	experiment_file_name_avg_reward = 'randomseed_' + str(random_seed) + '_avg_reward'
	path_to_save_avg_reward = log_dir + os.sep + experiment_file_name_avg_reward + '.npz'

	experiment_file_name_total_episodes = 'randomseed_' + str(random_seed) + '_total_episodes'
	path_to_save_total_episodes = log_dir + os.sep + experiment_file_name_total_episodes + '.npz'

	experiment_file_name_task_completion = 'randomseed_' + str(random_seed) + '_task_completion_curr'
	path_to_save_task_completion = log_dir + os.sep + experiment_file_name_task_completion + '.npz'

	# experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
	# path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'

	# experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
	# path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'

	# experiment_file_name_final_avg_reward = 'randomseed_' + str(random_seed) + '_final_avg_reward'
	# path_to_save_final_avg_reward = log_dir + os.sep + experiment_file_name_final_avg_reward + '.npz'

	np.savez_compressed(path_to_save_total_timesteps, curriculum_timesteps = total_timesteps_array)
	# np.delete(total_timesteps_array)

	np.savez_compressed(path_to_save_total_reward, curriculum_reward = total_reward_array)
	# np.delete(total_reward_array)

	np.savez_compressed(path_to_save_avg_reward, curriculum_avg_reward = avg_reward_array)
	# np.delete(avg_reward_array)

	np.savez_compressed(path_to_save_total_episodes, curriculum_episodes = total_episodes_arr)
	# np.delete(total_episodes_arr)

	np.savez_compressed(path_to_save_task_completion, task_completion_curr = task_completion_arr)

	# np.savez_compressed(path_to_save_final_timesteps, final_timesteps = final_timesteps_array)
	# # np.delete(final_timesteps_array)

	# np.savez_compressed(path_to_save_final_reward, final_reward = final_reward_array)
	# # final_reward_array.cler()

	# np.savez_compressed(path_to_save_final_avg_reward, final_avg_reward = final_avg_reward_array)
	# # np.delete(final_avg_reward_array)