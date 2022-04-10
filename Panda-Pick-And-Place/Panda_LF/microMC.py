import numpy as np
import time

# import our dqn agent
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt


REWARD_STEP = -1
REWARD_DONE = 1000

"""
grid key:
0:	empty space
1: 	wall
2:	tree
3:	rock
4:	craft table

action keys:
W:	go forward
A:	turn left
D:	turn right
U:	use/break block (must be facing tree or rock)
C: 	craft (must be facing craft table and have at least 2 wood and 1 stone)

use action key:
tree -> add 1 wood
rock -> add 1 stone

"""



class LidarSensor(object):
	def __init__(self,increment):
		
		self.angle_increment = increment # e.g., 2pi/10 if we want 10 beams
		self.sense_range = 10
		
		
	def sense(self,mc_env):
		num_obj_types = len(mc_env.object_types) # this includes empty square
		
		turn_offset = 0
		
		if mc_env.agent_dir == 'W':
			turn_offset -= np.pi/2
		elif mc_env.agent_dir == 'S':
			turn_offset -= 2*np.pi/2
		elif mc_env.agent_dir == 'E':
			turn_offset -= 3*np.pi/2
		
		current_angle = 0 + turn_offset
		
		lidar_readings = []
		
		#print("Agent location: "+str(mc_env.agent_x)+","+str(mc_env.agent_y))
		
		while True:
			#print("Beam at angle = "+str(current_angle))
			
			beam_i = np.zeros(num_obj_types) # we get 1 value per object, excluding empty square
			# print("beam length: ", len(beam_i))
		 # shoot beam
			for r in range(1,self.sense_range):
				# calc x and y position of beam at length r relative to agent
				x = mc_env.agent_x + np.round(r*np.cos(current_angle))
				y = mc_env.agent_y + np.round(r*np.sin(current_angle))
				obj_xy = mc_env.grid[int(x)][int(y)]
				
				if not obj_xy == 0: # if square is not empty
					
					sensor_value = float(self.sense_range - r)/float(self.sense_range)
					
					#print(str(int(x))+","+str(int(y))+","+str(int(obj_xy))+","+str(sensor_value))
					beam_i[obj_xy-1]=sensor_value
					#print beam_i
					
					break
			
			for k in range(0,len(beam_i)):
				lidar_readings.append(beam_i[k])
			#print lidar_readings	
			
			current_angle += self.angle_increment
			
			if current_angle >= 2*np.pi + turn_offset:
				break

		# print("lidar readings are:", lidar_readings)
		# print("len lidar readings: ", len(lidar_readings))
		return lidar_readings


class microMC(object):
	
	# constructor
	def __init__(self, width, height,random_seed):
		
		#np.random.seed(random_seed)
		
		self.object_types = [1, 2, 3, 4] # we have 4 objects: wall, tree, rock, and craft table
		
		rows, cols = (width, height) 
		
		self.width = width
		self.height = height
		self.grid = [[0 for i in range(cols)] for j in range(rows)] 
		
		# how many trees and rocks
		n_trees = int(width*height/20)
		n_rocks = int(width*height/40)

		# n_trees = 0
		# n_rocks = 0
		
		# fill in walls with 1s
		for i in range(0,self.height):
			for j in range(0,self.width):
				if i == 0 or i == self.height-1 or j == 0 or j == self.width-1:
					self.grid[j][i] = 1
					
		# create random trees
		for k in range(0,n_trees):
			x_k = np.random.randint(self.width-2)+1
			y_k = np.random.randint(self.height-2)+1
			self.grid[x_k][y_k]=2
			
			
		# create random rocks
		for k in range(0,n_rocks):
			x_k = np.random.randint(self.width-2)+1
			y_k = np.random.randint(self.height-2)+1
			self.grid[x_k][y_k]=3
			
		# create crafting table
		while True:
			x_k = np.random.randint(self.width-4)+2
			y_k = np.random.randint(self.height-4)+2
			if self.grid[x_k][y_k] == 0:
				self.grid[x_k][y_k] = 4
				break
		
		# initialize agent position and inventory
		while True:
			x_k = np.random.randint(self.width-4)+2
			y_k = np.random.randint(self.height-4)+2
			if self.grid[x_k][y_k] == 0:
				self.agent_x = x_k
				self.agent_y = y_k
				self.agent_dir = 'N' # start facing north
				break
		
		self.inventory = dict([('wood', 0), ('stone', 0),('pogo',0)])
		
		
	def toString(self):
		out_str = ''
		for i in range(0,self.height):
			for j in range(0,self.width):
				if self.agent_x == j and self.agent_y == i:
					if self.agent_dir == 'N':
						out_str += '^'
					elif self.agent_dir == 'S':
						out_str += 'v'
					elif self.agent_dir == 'E':
						out_str += '>'
					elif self.agent_dir == 'W':
						out_str += '<'
				elif self.grid[j][i] == 1: # obstacle/wall
					out_str += '#'
				elif self.grid[j][i] == 2: # tree
					out_str += 'T'
				elif self.grid[j][i] == 3: # rocks
					out_str += 'R'
				elif self.grid[j][i] == 4: # craft table
					out_str += 'C'
				elif self.grid[j][i] == 0: # free space
					out_str += ' '
				
				#out_str += str(self.grid[j][i])
				out_str += ' '
			out_str += '\n'
		
		out_str += '\ninventory:\t' + str(self.inventory)
		
		return out_str
	
	def getFacingXY(self): # get the x y position in front of the agent
		# compute the target position in front of the agent
		target_x = self.agent_x
		target_y = self.agent_y
		
		if self.agent_dir == 'N':
			target_y -= 1
		elif self.agent_dir == 'W':
			target_x -= 1
		elif self.agent_dir == 'E':
			target_x += 1
		elif self.agent_dir == 'S':
			target_y += 1
		return [target_x,target_y]
	
	def execute_action(self, action):
		reward = REWARD_STEP
		done = False
		
		# first, process turn actions
		if action == 'A': # turn right
			if self.agent_dir == 'N':
				self.agent_dir = 'W'
			elif self.agent_dir == 'W':
				self.agent_dir = 'S'
			elif self.agent_dir == 'S':
				self.agent_dir = 'E'
			elif self.agent_dir == 'E':
				self.agent_dir = 'N'
		elif action == 'D': # turn left
			if self.agent_dir == 'N':
				self.agent_dir = 'E'
			elif self.agent_dir == 'W':
				self.agent_dir = 'N'
			elif self.agent_dir == 'S':
				self.agent_dir = 'W'
			elif self.agent_dir == 'E':
				self.agent_dir = 'S'
		elif action == 'W': # go forward
			
			# compute the target position in front of the agent
			[target_x, target_y] = self.getFacingXY()
			
			if self.grid[target_x][target_y] == 0: # if target position is empty, move
				self.agent_x = target_x
				self.agent_y = target_y
		elif action == 'U': # use / break block
			# compute the target position in front of the agent
			[target_x, target_y] = self.getFacingXY()
				
			if self.grid[target_x][target_y] == 2: # if tree in front
				self.grid[target_x][target_y] = 0 # we clear the tree
				self.inventory['wood'] += 1
				if self.inventory['wood'] <= 2:
					reward = 10 # learn to chop wood if needed
			elif self.grid[target_x][target_y] == 3: # if rock in front
				self.grid[target_x][target_y] = 0 # we clear the tree
				self.inventory['stone'] += 1
				if self.inventory['stone'] <= 1:
					reward = 10 # learn to chop stone if needed
		elif action == 'C': # craft -- need 2 wood and 1 rock
			[target_x, target_y] = self.getFacingXY()
			if self.grid[target_x][target_y] == 4: # if craft in front
				if self.inventory['wood'] >= 2 and self.inventory['stone'] >= 1:
					self.inventory['pogo'] += 1
					self.inventory['wood'] -= 2
					self.inventory['stone'] -= 1
					done = True
					reward = REWARD_DONE
		return [done, reward]
			
			
def main_dqn():
	# environment
	random_seed = 10
	env = microMC(10,10,random_seed)
	
	# sensor
	sensor = LidarSensor(np.pi/8)
	
	# policy network
	actionCnt = 5
	D = 8 * 2 * len(env.object_types) + 2 # how many input neurons
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	
	agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	agent.set_explore_epsilon(MAX_EPSILON)
	
	action_space = ['W','A','D','U','C']
	
	t_step = 0
	episode = 0
	t_limit = 100
	reward_sum = 0
	reward_arr = []
	avg_reward = []
	while True:
		#print env.toString()
		
		# get obseration from sensor
		obs = sensor.sense(env)
	
		# add inventory observation
		obs.append(env.inventory['wood'])
		obs.append(env.inventory['stone'])
	
		# construct input x 
		x = np.asarray(obs)
		# print("observations are: ", x)
		# print("length obs: ", len(x))
		# act 
		a = agent.process_step(x,True)
		#print("Action at t="+str(t_step)+" is "+action_space[a])
		
		[done,reward] = env.execute_action(action_space[a])
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
			avg_reward.append(np.mean(reward_arr[-10:]))
			# print("avg_reward is: ", reward_arr)			
			done = True
			t_step = 0
			agent.finish_episode()
		
			# update after every episode
			agent.update_parameters()
		
			# reset environment
			episode += 1
			env = microMC(10,10,episode) # this is a bug causing memory leak, ideally env should have a reset function
			reward_sum = 0
	
			# quit after some number of episodes
			if episode > 12000:
				# print("avg reward is: ", avg_reward)
				plt.plot(avg_reward)
				plt.savefig('avg_reward_jivko2.png')
				break


# def main_teleop():

# 	env = microMC(20,10,5)
# 	print env.toString()
# 	sensor = LidarSensor(np.pi/8)


# 	while True:

# 		obs = sensor.sense(env)
# 		action = raw_input()
# 		[done,reward] = env.execute_action(action)
# 		print str(done)+"\t"+str(reward)
# 		print env.toString()

# 	print("Hello World!")

if __name__ == "__main__":
	main_dqn()
