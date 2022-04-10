import os
import sys

import gym
import time
import numpy as np
import gym_novel_gridworlds

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt



def CheckTrainingDoneCallback(reward_array, done_array, env):

	done_cond = False
	reward_cond = False
	if len(done_array) > 30:
		if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
			if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
				done_cond = True

		if done_cond == True:
			if env < 3:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True
			else:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True

		if done_cond == True and reward_cond == True:
			return 1
		else:
			return 0
	else:
		return 0



if __name__ == "__main__":

	no_of_environmets = 4

	width_array = [4,5,6,8]
	height_array = [4,5,6,8]
	length_array = [4,5,6,8]
	goal_env_array = [0,1,2,2]

	total_timesteps_array = []
	total_reward_array = []
	avg_reward_array = []
	final_timesteps_array = []
	final_reward_array = []
	final_avg_reward_array = []
	curr_task_completion_array = []
	final_task_completion_array = []

	actionCnt = 7
	# D = 37 #8 beams x 4 items lidar + 5 inventory items
	NUM_HIDDEN = 16
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	# agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	# agent.set_explore_epsilon(MAX_EPSILON)
	action_space = ['W','A','D','U','C']
	total_episodes_arr = []

	for i in range(no_of_environmets):
		# print("Environment: ", i)
		# i = 2

		width = width_array[i]
		height = height_array[i]
		length = length_array[i]
		goal = goal_env_array[i]

		env_id = 'NovelGridworld-v1'
		env = gym.make(env_id, map_width = width, map_height = height, map_length = length, goal_env = goal)
		D = len(env.low)

		if i == 0:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
		else:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
			agent.load_model(0,0,i-1)
			agent.reset()
			print("loaded model")


		
		t_step = 0
		episode = 0
		t_limit = 100
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
					curr_task_completion_array.append(1)
				elif t_step > t_limit:
					done_arr.append(0)
					curr_task_completion_array.append(0)
				
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
				if episode > 120000 or env_flag == 1:

					agent.save_model(0,0,i)
					total_episodes_arr.append(episode)

					break

	print("Total epsiode array is: ", total_episodes_arr)


	actionCnt = 5
	D = 37 #8 beams x 4 items lidar + 5 inventory items
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	action_space = ['W','A','D','U','C']
	# total_episodes_arr = []

	for k in range(1):
		# print("Environment: ", i)
		i = 3

		width = width_array[i]
		height = height_array[i]
		length = length_array[i]
		goal = goal_env_array[i]

		env_id = 'NovelGridworld-v1'
		env = gym.make(env_id, map_width = width, map_height = height, map_length = length, goal_env = goal)
		D = len(env.low)

		agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
		agent.set_explore_epsilon(MAX_EPSILON)
			
		t_step = 0
		episode = 0
		t_limit = 100
		reward_sum = 0
		reward_arr = []
		avg_reward = []
		done_arr = []
		env_flag = 0

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
				if done == True:
					done_arr.append(1)
					final_task_completion_array.append(1)
				elif t_step > t_limit:
					done_arr.append(0)
					final_task_completion_array.append(0)
				
				print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
				# print("epsilon is: ", MAX_EPSILON)
				reward_arr.append(reward_sum)
				avg_reward.append(np.mean(reward_arr[-40:]))

				final_timesteps_array.append(t_step)
				final_reward_array.append(reward_sum)
				final_avg_reward_array.append(np.mean(reward_arr[-40:]))

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

				# env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)
				env_flag = 0
		
				# quit after some number of episodes
				if episode > 120000 or env_flag == 1:

					agent.save_model(1,0,0)

					if env_flag == 1:
						print("ENV LEARNT after episodes: ", episode)
						time.sleep(5)

					# plt.plot(avg_reward)
					# plt.savefig('avg_reward_jivko9.png')
					break


	log_dir = 'logs_' + str(random_seed) 
	os.makedirs(log_dir, exist_ok = True)

	total_timesteps_array = np.asarray(total_timesteps_array).astype(np.uint16)
	print("size total_timesteps_array: ", total_timesteps_array.shape)
	
	total_reward_array = np.asarray(total_reward_array).astype(np.uint16)
	print("size total_reward_array: ", total_reward_array.shape)

	avg_reward_array = np.asarray(avg_reward_array).astype(np.uint16)
	print("size avg_reward_array: ", avg_reward_array.shape)

	total_episodes_arr = np.asarray(total_episodes_arr).astype(np.uint16)
	print("size total_episodes_arr: ", total_episodes_arr.shape)

	final_timesteps_array = np.asarray(final_timesteps_array).astype(np.uint16)
	print("size final_timesteps_array: ", final_timesteps_array.shape)

	final_reward_array = np.asarray(final_reward_array).astype(np.uint16)
	print("size final_reward_array: ", final_reward_array.shape)

	final_avg_reward_array = np.asarray(final_avg_reward_array).astype(np.uint16)
	print("size final_avg_reward_array: ", final_avg_reward_array.shape)

	curr_task_completion_arr = np.asarray(curr_task_completion_array)
	final_task_completion_arr = np.asarray(final_task_completion_array)

	experiment_file_name_total_timesteps = 'randomseed_' + str(random_seed) + '_total_timesteps'
	path_to_save_total_timesteps = log_dir + os.sep + experiment_file_name_total_timesteps + '.npz'

	experiment_file_name_total_reward = 'randomseed_' + str(random_seed) + '_total_reward'
	path_to_save_total_reward = log_dir + os.sep + experiment_file_name_total_reward + '.npz'

	experiment_file_name_avg_reward = 'randomseed_' + str(random_seed) + '_avg_reward'
	path_to_save_avg_reward = log_dir + os.sep + experiment_file_name_avg_reward + '.npz'

	experiment_file_name_total_episodes = 'randomseed_' + str(random_seed) + '_total_episodes'
	path_to_save_total_episodes = log_dir + os.sep + experiment_file_name_total_episodes + '.npz'

	experiment_file_name_task_completion = 'randomseed_' + str(random_seed) + '_task_completion_curr'
	path_to_save_curr_task_completion = log_dir + os.sep + experiment_file_name_task_completion + '.npz'

	experiment_file_name_final_task_completion = 'randomseed_' + str(random_seed) + '_task_completion_final'
	path_to_save_final_task_completion = log_dir + os.sep + experiment_file_name_final_task_completion + '.npz'

	experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
	path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'

	experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
	path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'

	experiment_file_name_final_avg_reward = 'randomseed_' + str(random_seed) + '_final_avg_reward'
	path_to_save_final_avg_reward = log_dir + os.sep + experiment_file_name_final_avg_reward + '.npz'

	np.savez_compressed(path_to_save_total_timesteps, curriculum_timesteps = total_timesteps_array)
	# np.delete(total_timesteps_array)

	np.savez_compressed(path_to_save_total_reward, curriculum_reward = total_reward_array)
	# np.delete(total_reward_array)

	np.savez_compressed(path_to_save_avg_reward, curriculum_avg_reward = avg_reward_array)
	# np.delete(avg_reward_array)

	np.savez_compressed(path_to_save_total_episodes, curriculum_episodes = total_episodes_arr)
	# np.delete(total_episodes_arr)
	np.savez_compressed(path_to_save_curr_task_completion, curr_task_completion = curr_task_completion_arr)

	np.savez_compressed(path_to_save_final_task_completion, final_task_completion = final_task_completion_arr)


	np.savez_compressed(path_to_save_final_timesteps, final_timesteps = final_timesteps_array)
	# np.delete(final_timesteps_array)

	np.savez_compressed(path_to_save_final_reward, final_reward = final_reward_array)
	# final_reward_array.cler()

	np.savez_compressed(path_to_save_final_avg_reward, final_avg_reward = final_avg_reward_array)
	# np.delete(final_avg_reward_array)