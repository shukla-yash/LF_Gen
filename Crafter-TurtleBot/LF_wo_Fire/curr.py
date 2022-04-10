import os
import sys

import gym
import time
import numpy as np
import gym_novel_gridworlds
import copy
# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt


# This function is a callback that checks if the environment is learnt. 
def CheckTrainingDoneCallback(reward_array, done_array, env):

	done_cond = False
	reward_cond = False
	if len(done_array) > 30:
		if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
			if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
				done_cond = True

		if done_cond == True:
			if env == 0:
				if np.mean(reward_array[-10:]) > 970: # Reward threshold on the final environment is higher because no reward shaping
					reward_cond = True
			else:
				if np.mean(reward_array[-10:]) > 950: # Reward threshold on the final environment is lower because no reward shaping
					reward_cond = True

		if done_cond == True and reward_cond == True:
			return 1
		else:
			return 0
	else:
		return 0


# This function returns the width, height, items in the env, initial inventory, and goal
def parameterizing_function(navigation, breaking, crafting, prev_width, prev_height):

	if navigation == 1 and breaking == 1:
		type_of_env = 2 
	if breaking == 1 and crafting == 1:
		type_of_env = 0
	if navigation == 1 and crafting == 1:
		type_of_env = 1
	if navigation == 1 and breaking < 1 and crafting < 1:
		type_of_env = np.random.randint(1,3)
	if breaking == 1 and navigation < 1 and crafting < 1:
		temp = np.random.randint(0,2)
		if temp == 0:
			type_of_env = 0
		if temp == 1:
			type_of_env = 2
	if crafting == 1 and breaking < 1 and navigation < 1:
		type_of_env = np.random.randint(0,2)
	if navigation < 1 and breaking < 1 and crafting < 1:
		type_of_env = np.random.randint(0,3)



	while True:
		width = np.random.randint(8,12)
		if width > prev_width:
			break
		elif prev_width == 11:
			width = 11
			break

	while True:
		height = np.random.randint(8,12)
		if height > prev_height:
			break
		elif prev_height == 11:
			height = 11
			break

	if type_of_env == 0:
		object_present = np.random.randint(0,3) # 0 -> Tree, 1 -> Rock, 2 -> Crafting Table
		if object_present == 0:
			total_trees =1 
			starting_trees = 0
			no_trees = 1
			total_rocks = 0
			no_rocks = 0
			starting_rocks = 0
			crafting_table = 0
		if object_present == 1:
			total_trees = 0 
			starting_trees = 0
			no_trees = 0
			total_rocks = 1
			no_rocks = 1
			starting_rocks = 0
			crafting_table = 0
		if object_present == 2:
			total_trees = 2 
			starting_trees = 2
			no_trees = 0
			total_rocks = 1
			no_rocks = 0
			starting_rocks = 1
			crafting_table = 1

	if type_of_env == 1:

		total_trees = np.random.randint(0,3)

		while True:
			total_trees = np.random.randint(0,3)
			total_rocks = np.random.randint(0,2)
			if total_trees > 0 or total_rocks > 0:
				if total_trees < 2 or total_rocks < 1:
					break

		while True:
			starting_trees = np.random.randint(0,total_trees + 1)
			starting_rocks = np.random.randint(0,total_rocks + 1)
			if starting_trees == 0:
				if starting_rocks is not total_rocks:
					break
			if starting_rocks == 0:
				if starting_trees is not total_trees:
					break

		no_trees = total_trees - starting_trees
		no_rocks = total_rocks - starting_rocks
		crafting_table = 0

	if type_of_env == 2:
		total_trees = 2
		starting_trees = np.random.randint(0, 3)
		no_trees = total_trees - starting_trees

		total_rocks = 1
		if total_rocks == 1:
			starting_rocks = 0
		else:
			starting_rocks = np.random.randint(0, 2)
		no_rocks = total_rocks - starting_rocks
		crafting_table = 1

	return width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env

if __name__ == "__main__":

	no_of_environmets = 15
	beam_search_width = 4
	curriculum_breadth = 3

	width_array = []
	height_array = []
	no_trees_array = []
	no_rocks_array = []
	crafting_table_array = []
	starting_trees_array = []
	starting_rocks_array = []
	type_of_env_array = []
	total_episodes_array_0 = []
	total_episodes_array_1 = []
	total_episodes_array_2 = []
	final_episodes_array = []
	
	curriculum_params = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0': 0}] for _ in range(beam_search_width)]


	actionCnt = 5
	D = 37 #8 beams x 4 items lidar + 5 inventory items
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	action_space = ['W','A','D','U','C']


	# Below for loop is for the 0th task
	for i in range(no_of_environmets):

		width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env = parameterizing_function(navigation = 0, breaking = 0, crafting = 0, prev_width = 0, prev_height = 0)
		width_array.append(width)
		height_array.append(height)
		no_trees_array.append(no_trees)
		no_rocks_array.append(no_rocks)
		crafting_table_array.append(crafting_table)
		starting_trees_array.append(starting_trees)
		starting_rocks_array.append(starting_rocks)
		type_of_env_array.append(type_of_env)


		agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
		agent.set_explore_epsilon(MAX_EPSILON)

		final_status = False
		env_id = 'NovelGridworld-v0'

		env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'stone_axe':0},
			initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'stone_axe':0}, goal_env = type_of_env, is_final = final_status)
		
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
				elif t_step > t_limit:
					done_arr.append(0)
				
				print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
				# print("epsilon is: ", MAX_EPSILON)
				reward_arr.append(reward_sum)
				avg_reward.append(np.mean(reward_arr[-40:]))
				# print("avg_reward is: ", reward_arr)			
				done = True
				t_step = 0
				agent.finish_episode()
			
				# update after every episode
				agent.update_parameters()
			
				# reset environment
				episode += 1
				# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
				# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'stone_axe':0}, \
				# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'stone_axe':0}, goal_env = 2)
				env.reset()
				reward_sum = 0

				env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, 0)
				# env_flag = 0
		
				# quit after some number of episodes
				if episode > 50000 or env_flag == 1:

					agent.save_model(0,0,i)
					total_episodes_array_0.append(episode)

					break

	temp_array = total_episodes_array_0.copy()
	temp_array.sort()
	elements_list = [total_episodes_array_0.index(temp_array[i]) for i in range(beam_search_width)]

	for beam_no in range(beam_search_width):
		curriculum_params[beam_no][0]['width0'] = width_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['height0'] = height_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['no_trees'] = no_trees_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['no_rocks'] = no_rocks_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['starting_trees'] = starting_trees_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['starting_rocks'] = starting_rocks_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['crafting_table'] = crafting_table_array[elements_list[beam_no]]
		curriculum_params[beam_no][0]['type_of_env0'] = type_of_env_array[elements_list[beam_no]]
		if type_of_env_array[elements_list[beam_no]] == 0:
			curriculum_params[beam_no][0]['navigation'] += 1
		if type_of_env_array[elements_list[beam_no]] == 1:
			curriculum_params[beam_no][0]['breaking'] += 1
		if type_of_env_array[elements_list[beam_no]] == 2:
			curriculum_params[beam_no][0]['crafting'] += 1

		agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
		agent.set_explore_epsilon(MAX_EPSILON)
		agent.load_model(0,0,elements_list[beam_no])
		for env_no in range(no_of_environmets):
			agent.save_model(0,beam_no,env_no)

	width_after = []
	height_after = []
	no_trees_after = []
	no_rocks_after = []
	starting_trees_after = []
	starting_rocks_after = []
	crafting_table_after = []
	type_of_env_after = []

	curriculum_params_1 = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0' : 0,\
	'width1': 0, 'height1': 0, 'no_trees1': 0, 'no_rocks1': 0, 'starting_trees1': 0, 'starting_rocks1': 0, 'crafting_table1': 0, 'type_of_env1' : 0}] for _ in range(beam_search_width)]

	curriculum_params_2 = [[{'width0': 0, 'height0': 0, 'no_trees': 0, 'no_rocks': 0, 'starting_trees': 0, 'starting_rocks': 0, 'crafting_table': 0, 'navigation': 0, 'breaking': 0, 'crafting': 0, 'type_of_env0' : 0, \
	'width1': 0, 'height1': 0, 'no_trees1': 0, 'no_rocks1': 0, 'starting_trees1': 0, 'starting_rocks1': 0, 'crafting_table1': 0, 'type_of_env1' : 0, \
	'width2': 0, 'height2': 0, 'no_trees2': 0, 'no_rocks2': 0, 'starting_trees2': 0, 'starting_rocks2': 0, 'crafting_table2': 0, 'type_of_env2' : 0}] for _ in range(beam_search_width)]


	# Below for loop is for the 1st and 2nd task
	for curriculum_number in range(1, curriculum_breadth):
		width_after.clear()
		height_after.clear()
		no_trees_after.clear()
		no_rocks_after.clear()
		starting_trees_after.clear()
		starting_rocks_after.clear()
		crafting_table_after.clear()
		type_of_env_after.clear()

		for beam_number in range(beam_search_width):
			for environment_number in range(no_of_environmets):

				print("Curriculum number: ", curriculum_number)
				print("Beam number: ", beam_number)
				print("Environment number: ", environment_number)

				if curriculum_number == 1:
					no_navigation = curriculum_params[beam_number][0]['navigation']
					no_breaking = curriculum_params[beam_number][0]['breaking']
					no_crafting = curriculum_params[beam_number][0]['crafting']
					prev_width = curriculum_params[beam_number][0]['width' + str(curriculum_number - 1)]
					prev_height = curriculum_params[beam_number][0]['height' + str(curriculum_number - 1)]
				if curriculum_number == 2:
					no_navigation = curriculum_params_1[beam_number][0]['navigation']
					no_breaking = curriculum_params_1[beam_number][0]['breaking']
					no_crafting = curriculum_params_1[beam_number][0]['crafting']
					prev_width = curriculum_params_1[beam_number][0]['width' + str(curriculum_number - 1)]
					prev_height = curriculum_params_1[beam_number][0]['height' + str(curriculum_number - 1)]

				width, height, no_trees, no_rocks, crafting_table, starting_trees, starting_rocks, type_of_env = parameterizing_function(navigation = no_navigation, breaking = no_breaking, crafting = no_crafting, prev_width = prev_width, prev_height = prev_height)
				width_after.append(width)
				height_after.append(height)
				no_trees_after.append(no_trees)
				no_rocks_after.append(no_rocks)
				crafting_table_after.append(crafting_table)
				starting_trees_after.append(starting_trees)
				starting_rocks_after.append(starting_rocks)
				type_of_env_after.append(type_of_env)

				print("here")

				agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
				agent.set_explore_epsilon(MAX_EPSILON)

				agent.load_model(curriculum_number-1, beam_number, environment_number)

				final_status = False
				env_id = 'NovelGridworld-v0'
				env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table, 'stone_axe':0},
					initial_inventory = {'wall': 0, 'tree': starting_trees, 'rock': starting_rocks, 'crafting_table': 0, 'stone_axe':0}, goal_env = type_of_env, is_final = final_status)

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
						elif t_step > t_limit:
							done_arr.append(0)
						
						print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
						# print("epsilon is: ", MAX_EPSILON)
						reward_arr.append(reward_sum)
						avg_reward.append(np.mean(reward_arr[-40:]))
						# print("avg_reward is: ", reward_arr)			
						done = True
						t_step = 0
						agent.finish_episode()

						# update after every episode
						agent.update_parameters()

						# reset environment
						episode += 1
						# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
						# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'stone_axe':0}, \
						# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'stone_axe':0}, goal_env = 2)
						env.reset()
						reward_sum = 0

						env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, 0)
						# env_flag = 0

						# quit after some number of episodes
						if episode > 50000 or env_flag == 1:

							agent.save_model(curriculum_number-1,beam_no,environment_number)
							if curriculum_number == 1:
								total_episodes_array_1.append(episode)
							elif curriculum_number == 2:
								total_episodes_array_2.append(episode)
							break

		elements_list.clear()
		if curriculum_number == 1:
			temp_array = total_episodes_array_1.copy()
			temp_array.sort()
			elements_list = [total_episodes_array_1.index(temp_array[i]) for i in range(beam_search_width)]
		elif curriculum_number == 2:
			temp_array = total_episodes_array_2.copy()
			temp_array.sort()
			elements_list = [total_episodes_array_2.index(temp_array[i]) for i in range(beam_search_width)]

		for i in range(beam_search_width):

			quotient = elements_list[i] // no_of_environmets
			remainder = elements_list[i] % no_of_environmets

			if curriculum_number == 1:
				curriculum_params_1[i] = copy.deepcopy(curriculum_params[quotient])
				curriculum_params_1[i][0].update({'width'+str(curriculum_number) : width_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'height'+str(curriculum_number) : height_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'no_trees'+str(curriculum_number) : no_trees_after[elements_list[i]]}) 
				curriculum_params_1[i][0].update({'no_rocks'+str(curriculum_number) : no_rocks_after[elements_list[i]]}) 
				curriculum_params_1[i][0].update({'starting_trees'+str(curriculum_number) : starting_trees_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'starting_rocks'+str(curriculum_number) : starting_rocks_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'crafting_table'+str(curriculum_number) : crafting_table_after[elements_list[i]]})
				curriculum_params_1[i][0].update({'type_of_env'+str(curriculum_number) : type_of_env_after[elements_list[i]]})

				if type_of_env_after[elements_list[i]] == 0:
					curriculum_params_1[i][0]['navigation'] += 1
				elif type_of_env_after[elements_list[i]] == 1:
					curriculum_params_1[i][0]['breaking'] += 1
				elif type_of_env_after[elements_list[i]] == 2:
					curriculum_params_1[i][0]['crafting'] += 1

			if curriculum_number == 2:
				curriculum_params_2[i] = copy.deepcopy(curriculum_params_1[quotient])
				curriculum_params_2[i][0].update({'width'+str(curriculum_number) : width_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'height'+str(curriculum_number) : height_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'no_trees'+str(curriculum_number) : no_trees_after[elements_list[i]]}) 
				curriculum_params_2[i][0].update({'no_rocks'+str(curriculum_number) : no_rocks_after[elements_list[i]]}) 
				curriculum_params_2[i][0].update({'starting_trees'+str(curriculum_number) : starting_trees_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'starting_rocks'+str(curriculum_number) : starting_rocks_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'crafting_table'+str(curriculum_number) : crafting_table_after[elements_list[i]]})
				curriculum_params_2[i][0].update({'type_of_env'+str(curriculum_number) : type_of_env_after[elements_list[i]]})

				if type_of_env_after[elements_list[i]] == 0:
					curriculum_params_2[i][0]['navigation'] += 1
				elif type_of_env_after[elements_list[i]] == 1:
					curriculum_params_2[i][0]['breaking'] += 1
				elif type_of_env_after[elements_list[i]] == 2:
					curriculum_params_2[i][0]['crafting'] += 1

			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
			agent.load_model(curriculum_number-1, quotient, remainder)

			if not curriculum_number == curriculum_breadth-1:
				for j in range(no_of_environmets):
					agent.save_model(curriculum_number, i, j)
			elif curriculum_number == curriculum_breadth-1:
				agent.save_model(curriculum_number,i,0)


	# Below for loop is for the final task (No reward shaping)
	for final_env in range(beam_search_width):
		print("Final env! ", final_env)

		agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
		agent.set_explore_epsilon(MAX_EPSILON)
		agent.load_model(curriculum_breadth-1, final_env, 0)

		final_status = True
		env_id = 'NovelGridworld-v0'
		env = gym.make(env_id, map_width = 12, map_height = 12, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'stone_axe':0},
			initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'stone_axe':0}, goal_env = 2, is_final = final_status)
		
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
				elif t_step > t_limit:
					done_arr.append(0)
				
				print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")
				# print("epsilon is: ", MAX_EPSILON)
				reward_arr.append(reward_sum)
				avg_reward.append(np.mean(reward_arr[-40:]))
				# print("avg_reward is: ", reward_arr)			
				done = True
				t_step = 0
				agent.finish_episode()
			
				# update after every episode
				agent.update_parameters()
			
				# reset environment
				episode += 1
				# env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
				# env.reset(map_width = 10, map_height = 10, items_quantity = {'tree': 4, 'rock': 2, 'crafting_table': 1, 'stone_axe':0}, \
				# 	initial_inventory = {'wall': 0, 'tree': 0, 'rock': 0, 'crafting_table': 0, 'stone_axe':0}, goal_env = 2)
				env.reset()
				reward_sum = 0

				env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, 1)
				# env_flag = 0
		
				# quit after some number of episodes
				if episode > 80000 or env_flag == 1:

					agent.save_model(curriculum_breadth,final_env,0)
					final_episodes_array.append(episode)
					break

	print("Curriculum params: ", curriculum_params_2)
	print("\n")
	print("Time taken for the final environment is: ", final_episodes_array)