import numpy as np
import matplotlib.pyplot as plt
import os

rolling_window_eps = 3000
num_random_seeds = 1
seed_arr = [1]
log_dir = 'logs'
data_total_timesteps = [[] for _ in range(num_random_seeds)]
data_total_reward = [[] for _ in range(num_random_seeds)]
data_avg_reward = [[] for _ in range(num_random_seeds)]
data_total_episodes = [[] for _ in range(num_random_seeds)]
data_final_timesteps = [[] for _ in range(num_random_seeds)]
data_final_reward = [[] for _ in range(num_random_seeds)]
data_final_avg_reward = [[] for _ in range(num_random_seeds)]

for i in range(num_random_seeds):
	log_dir = 'logs_' + str(seed_arr[i])
	random_seed = seed_arr[i]
	experiment_file_name_total_timesteps = 'randomseed_' + str(random_seed) + '_total_timesteps'
	path_to_save_total_timesteps = log_dir + os.sep + experiment_file_name_total_timesteps + '.npz'
	temp = np.load(path_to_save_total_timesteps)
	print("size: ", temp['curriculum_timesteps'].shape)
	data_total_timesteps[i] = temp['curriculum_timesteps']

	experiment_file_name_total_reward = 'randomseed_' + str(random_seed) + '_total_reward'
	path_to_save_total_reward = log_dir + os.sep + experiment_file_name_total_reward + '.npz'
	temp = np.load(path_to_save_total_reward)
	data_total_reward[i] = temp['curriculum_reward']
	print("length data total reward: ", len(data_total_reward[i]))

	experiment_file_name_avg_reward = 'randomseed_' + str(random_seed) + '_avg_reward'
	path_to_save_avg_reward = log_dir + os.sep + experiment_file_name_avg_reward + '.npz'
	temp = np.load(path_to_save_avg_reward)
	data_avg_reward[i] = temp['curriculum_avg_reward']
	print("data_avg_reward: ", len(data_avg_reward[i]))


	experiment_file_name_total_episodes = 'randomseed_' + str(random_seed) + '_total_episodes'
	path_to_save_total_episodes = log_dir + os.sep + experiment_file_name_total_episodes + '.npz'
	temp = np.load(path_to_save_total_episodes)
	data_total_episodes[i] = temp['curriculum_episodes']
	print("data _total_episodes: ", data_total_episodes)


	experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
	path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'
	temp = np.load(path_to_save_final_timesteps)
	data_final_timesteps[i] = temp['final_timesteps']


	experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
	path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'
	temp = np.load(path_to_save_final_reward)
	data_final_reward[i] = temp['final_reward']


	experiment_file_name_final_avg_reward = 'randomseed_' + str(random_seed) + '_final_avg_reward'
	path_to_save_final_avg_reward = log_dir + os.sep + experiment_file_name_final_avg_reward + '.npz'
	temp = np.load(path_to_save_final_avg_reward)
	data_final_avg_reward[i] = temp['final_avg_reward']


total_reward_arr_shifted = [[] for _ in range(num_random_seeds)]
avg_reward_arr_shifted = [[] for _ in range(num_random_seeds)]
for i in range(num_random_seeds):
        data_total_timesteps[i] = data_total_timesteps[i].tolist()
        data_total_reward[i] = data_total_reward[i].tolist()
        data_final_timesteps[i] = data_final_timesteps[i].tolist()
        data_final_reward[i] = data_final_reward[i].tolist()
        data_final_avg_reward[i] = data_final_avg_reward[i].tolist()


final_reward_shifted = [[] for _ in range(num_random_seeds)]
curriculum_episodes = []
for i in range(num_random_seeds):
	curriculum_episodes.append(sum(data_total_episodes[i][:3]))
	total_reward_arr_shifted[i] = [np.nan] * curriculum_episodes[i]
	print("curriculum_eps: ", curriculum_episodes)
	print("length before: ", len(total_reward_arr_shifted[i]))
	print("extending length: ", len(data_total_reward[i][curriculum_episodes[i]:]))
	total_reward_arr_shifted[i].extend(data_total_reward[i][curriculum_episodes[i]:])
	print("length after: ", len(total_reward_arr_shifted[i]))
	avg_reward_arr_shifted[i] = [np.nan] * curriculum_episodes[i]
	avg_reward_arr_shifted[i].extend(data_avg_reward[i][curriculum_episodes[i]:])
	print("reward thing: ", len(avg_reward_arr_shifted[i]))

print("hereee00")

curriculum_shifted_reward_arr = []
curriculum_shifted_timestep_arr = []
final_reward_arr = []
final_timestep_arr = []

max_len = max(len(avg_reward_arr_shifted[i]) for i in range(len(avg_reward_arr_shifted)))
max_len_2 = max(len(data_final_avg_reward[i]) for i in range(len(data_final_avg_reward)))
max_len = max(max_len, max_len_2)

print("maxlen is: ", max_len)


for i in range(num_random_seeds):
	print(len(avg_reward_arr_shifted[i]))
	if len(avg_reward_arr_shifted[i]) < max_len:
		while len(avg_reward_arr_shifted[i]) < max_len:
			avg_reward_arr_shifted[i].append(np.nan)
			data_total_timesteps[i].append(0)

for i in range(num_random_seeds):
	if len(data_final_avg_reward[i]) < max_len:
		while len(data_final_avg_reward[i]) < max_len:
			data_final_avg_reward[i].append(np.nan)
			data_final_timesteps[i].append(0)			
# print("len 0:", len(avg_reward_arr_shifted[0]))
# print("len 1:", len(avg_reward_arr_shifted[1]))

for i in range(max_len):
	for j in range(len(data_final_avg_reward)):
		if data_final_avg_reward[j][i] > 1000:
			data_final_avg_reward[j][i] = 0

current_index_curr = 0
current_index_final = 0

std_dev_curr = []
std_dev_curr_arr = []
std_dev_final = []
std_dev_final_arr = []

for i in range(0, max_len - int(rolling_window_eps), int(rolling_window_eps/2)):
	print("i is: ", i)
	curr_sum = 0
	curr_avg = 0
	curr_count = 0
	final_sum = 0
	final_avg = 0
	final_count = 0
	curr_sum_timesteps = 0
	curr_avg_timesteps = 0
	final_sum_timesteps = 0
	final_avg_timesteps = 0

	# std_dev_curr.clear()
	std_dev_curr_arr.clear()
	# std_dev_final.clear()
	std_dev_final_arr.clear()

	loop_count = 0
	for j in range(int(rolling_window_eps)):
		for k in range(len(avg_reward_arr_shifted)):
			loop_count += 1
			curr_sum_timesteps += data_total_timesteps[k][i+j]
			# curr_avg_timesteps = curr_sum_timesteps / loop_count
			curr_avg_timesteps = curr_sum_timesteps

			if not np.isnan(avg_reward_arr_shifted[k][i+j]):
				curr_count += 1
				curr_sum += avg_reward_arr_shifted[k][i+j]
				curr_avg = curr_sum/curr_count
				std_dev_curr_arr.append(avg_reward_arr_shifted[k][i+j])
			if j == int(rolling_window_eps/2)-1 and k == len(avg_reward_arr_shifted)-1 and curr_count == 0:
				curriculum_shifted_reward_arr.append(np.nan)
				current_index_curr += curr_avg_timesteps
				curriculum_shifted_timestep_arr.append(current_index_curr)
				# std_dev = np.nan
				# std_dev_curr.append(std_dev)
			elif j == int(rolling_window_eps/2)-1 and k == len(avg_reward_arr_shifted)-1:
				curriculum_shifted_reward_arr.append(curr_avg)
				current_index_curr += curr_avg_timesteps
				curriculum_shifted_timestep_arr.append(current_index_curr)
				std_dev = np.std(std_dev_curr_arr, ddof = 1)
				std_dev_curr.append(std_dev)

		for k in range(len(data_final_avg_reward)):
			loop_count += 1
			final_sum_timesteps += data_final_timesteps[k][i+j]
			# final_avg_timesteps = final_sum_timesteps / loop_count
			final_avg_timesteps = final_sum_timesteps 

			if not np.isnan(data_final_avg_reward[k][i+j]):
				final_count += 1
				final_sum += data_final_avg_reward[k][i+j]
				final_avg = final_sum/final_count
				std_dev_final_arr.append(data_final_avg_reward[k][i+j])
			if j == int(rolling_window_eps/2)-1 and k == len(data_final_avg_reward)-1 and final_count == 0:
				final_reward_arr.append(np.nan)
				current_index_final += final_avg_timesteps
				final_timestep_arr.append(current_index_final)
				# std_dev_final_val = np.nan
				# std_dev_final.append(std_dev_final_val)
			elif j == int(rolling_window_eps/2)-1 and k == len(data_final_avg_reward)-1:
				final_reward_arr.append(final_avg)
				current_index_final += final_avg_timesteps
				final_timestep_arr.append(current_index_final)
				std_dev_final_val = np.std(std_dev_final_arr, ddof = 1)
				std_dev_final.append(std_dev_final_val)


max_latest_curr = np.max(curriculum_episodes)

# for i in range(28000):
# 	curriculum_shifted_reward_arr[i] = np.nan

print("len a:", len(curriculum_shifted_reward_arr))
print("len b:", len(final_reward_arr))

# print("X AXIS::: \n",std_dev_curr)

x_count = 0
x_count_f = 0



min_x = min(len(curriculum_shifted_timestep_arr), len(final_timestep_arr))

for i in range(min_x):
	if curriculum_shifted_timestep_arr[i] < 100000000:
		x_count += 1

for i in range(min_x):
	if final_timestep_arr[i] < 100000000:
		x_count_f += 1


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.loc':'lower right'})
plt.rcParams.update({'lines.markersize': 8})
plt.ylim([-900,900])


# plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'legend.loc':'lower right'})

plt.scatter(curriculum_shifted_timestep_arr[:x_count], curriculum_shifted_reward_arr[:x_count], marker = 'o', label = 'Curriculum')
plt.errorbar(curriculum_shifted_timestep_arr[:x_count], curriculum_shifted_reward_arr[:x_count], yerr = std_dev_curr[:x_count])
# plt.plot(curriculum_shifted_timestep_arr[:min_x-19], curriculum_shifted_reward_arr[:min_x-19], label = 'learning through curriculum')

plt.scatter(final_timestep_arr[:x_count_f], final_reward_arr[:x_count_f], marker = 'x',label = 'Baseline')
plt.errorbar(final_timestep_arr[:x_count_f], final_reward_arr[:x_count_f], yerr = std_dev_final[:x_count_f])
# plt.plot(final_timestep_arr[:min_x-19], final_reward_arr[:min_x-19], label = 'learning from scratch')

# plt.plot(curriculum_shifted_timestep_arr, curriculum_shifted_reward_arr, label = 'learning through curriculum')
plt.xlabel('Timesteps')
plt.ylabel('Average rewards')
plt.legend()
# plt.show()
plt.savefig("HF_HC_wo_Fire_2")
print("doing everything")
