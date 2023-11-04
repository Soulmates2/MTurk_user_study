import argparse
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import re
import sys


# parser = argparse.ArgumentParser(description='Analyze directory of MTurk 2AFC study results')

# # In a single HIT, how many comparisons did the Turker make?
# parser.add_argument('--n-real-comparisons-per-hit', type=int, default=50)
# # How many "vigilance tests" (i.e. 'obvious' comparisons to check if the Turker is paying attention) per HIT?
# parser.add_argument('--n-vigilance-tests-per-hit', type=int, default=5)
# # If a Turker gets below this percentage of vigilance tests correct, throw out their responses
# parser.add_argument('--vigilance-threshold', type=float, default=0.0)
# # Directory containing CSV files to analyze
# parser.add_argument('--in-dir', type=str)
# # Name of output file to write
# parser.add_argument('--out-filename', type=str, default='analysis.csv')

# args = parser.parse_args()

np.random.seed(2024)
n_real_comparisons_per_hit = 20
n_vigilance_tests_per_hit = 5
vigilance_threshold = 1.0
task = ''
in_dir = f'data/20231031_2d_1/result_csv/'
out_dir = f'data/20231031_2d_1/analysis/'
out_filename = f'data/20231031_2d_1/analysis/0_analysis.csv'
n_comparisons = n_real_comparisons_per_hit + n_vigilance_tests_per_hit
n_vigilance = n_vigilance_tests_per_hit
alpha=0.050

reject_line = "We would like to extend our deepest gratitude for your time and efforts taken to participate in our survey. However, the vigilance tests embedded within the survey are designed to ensure the accuracy and reliability of the data collected. These tests were unfortunately not passed with the necessary accuracy in your submission. With this in mind, we regret to inform you that we are unable to include your submission in our data set."


def analyze(loc, n_bootstrap_samples=10000, alpha=0.050):
	print(loc)

	df = pd.read_csv(loc)
    new_df = pd.DataFrame.copy()

	# Figure out how many hits there are by looking at one of the fields
	n_hits = len(df['Input.gt_side1'])

	# Keep track of how each worker does on vigilance tests
	worker_vigilance = defaultdict(int)
	for i in range(n_comparisons):
		gt_sides = df['Input.gt_side' + str(i+1)]
		choice = df['Answer.selection' + str(i+1)]
		img_left = df['Input.images_left' + str(i+1)]
		img_mid = df['Input.images_mid' + str(i+1)]
		img_right = df['Input.images_right' + str(i+1)]
		for j in range(n_hits):
			if ('vigilance' in img_left[j]) or ('vigilance' in img_mid[j]) or ('vigilance' in img_right[j]):
				this_gt_side = gt_sides[j]
				this_choice = choice[j]
				worker_vigilance[f"{df['HITId'][j]}/{df['WorkerId'][j]}"] += int(this_gt_side == this_choice)
			df['Answer.comments'][j] = ""
	# Print worker accuracy on vigilance tests:
	num_workers_passed = 0
	worker_ids = []
	for worker_id in worker_vigilance.keys():
		worker_vigilance[worker_id] = float(worker_vigilance[worker_id])/n_vigilance
		print('Worker ID, vigilance: ', worker_id, worker_vigilance[worker_id])
		if worker_vigilance[worker_id] >= vigilance_threshold:
			num_workers_passed += 1
		
	print("Num workers passed vigilance threshold: " + str(num_workers_passed))
	print(sorted(worker_ids))
	# Now record actual comparison stats
	value_a = []
	value_b = []
	value_c = []
	for i in range(n_comparisons):
		gt_sides = df['Input.gt_side' + str(i+1)]
		choice = df['Answer.selection' + str(i+1)]
		img_left = df['Input.images_left' + str(i+1)]
		img_mid = df['Input.images_mid' + str(i+1)]
		img_right = df['Input.images_right' + str(i+1)]
		for j in range(n_hits):
			# Skip workers that didn't pass enough vigilance tests
			if worker_vigilance[f"{df['HITId'][j]}/{df['WorkerId'][j]}"] < vigilance_threshold:
				df['Reject'][j] = reject_line
                continue
			# Only record values that don't come from vigilance tests
			if ('vigilance' in img_left[j]) or ('vigilance' in img_mid[j]) or ('vigilance' in img_right[j]):
				continue
			this_gt_side = gt_sides[j]
			this_choice = choice[j]
			value_a.append(int(this_gt_side == this_choice))

			if ('ARAP' in img_left[j]):
				this_arap = 'a'
			elif ('ARAP' in img_mid[j]):
				this_arap = 'b'
			elif ('ARAP' in img_right[j]):
				this_arap = 'c'
			else:
				raise ValueError("image doesn't contain ARAP")
			value_b.append(int(this_arap == this_choice))
			
			if ('DragDiffusion' in img_left[j]):
				this_dragdiff = 'a'
			elif ('DragDiffusion' in img_mid[j]):
				this_dragdiff = 'b'
			elif ('DragDiffusion' in img_right[j]):
				this_dragdiff = 'c'
			else:
				raise ValueError("image doesn't contain DragDiffusion")
			value_c.append(int(this_dragdiff == this_choice))
	

	# Convert to np array
	value_a = np.array(value_a)
	value_b = np.array(value_b)
	value_c = np.array(value_c)

	print(f'Ours: {np.mean(value_a)}')
	print(f'ARAP: {np.mean(value_b)}')
	print(f'DragDiffusion: {np.mean(value_c)}')
	print(f'Total: {np.mean(value_a) + np.mean(value_b) + np.mean(value_c)}')
	
	# Do bootstrap
	sample_a = []
	sample_b = []
	sample_c = []
	for _ in range(n_bootstrap_samples):
		# Sample values with replacement
		a_bootstrap_sample = np.random.choice(value_a, replace=True, size=len(value_a))
		sample_a.append(np.mean(a_bootstrap_sample))
		b_bootstrap_sample = np.random.choice(value_b, replace=True, size=len(value_b))
		sample_b.append(np.mean(b_bootstrap_sample))
		c_bootstrap_sample = np.random.choice(value_c, replace=True, size=len(value_c))
		sample_c.append(np.mean(c_bootstrap_sample))


	# samples = values
	sample_a = np.array(sample_a)
	mean_a = np.mean(sample_a)
	std_a = np.std(sample_a)
	# Compute confidence intervals
	# https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Methods_for_bootstrap_confidence_intervals
	# Using the first one (basic bootstrap) in above link
	low_a = 2 * mean_a - np.percentile(sample_a, 100 * (1 - alpha / 2.))
	high_a = 2 * mean_a - np.percentile(sample_a, 100 * (alpha / 2.))

	# Compare with the package for sanity check
	print('Ours mean, stdev, low, high: ', mean_a, std_a, low_a, high_a)
	print('Ours mean, conf interval: ', bs.bootstrap(value_a, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=value_a.size, num_iterations=n_bootstrap_samples))

	sample_b = np.array(sample_b)
	mean_b = np.mean(sample_b)
	std_b = np.std(sample_b)
	low_b = 2 * mean_b - np.percentile(sample_b, 100 * (1 - alpha / 2.))
	high_b = 2 * mean_b - np.percentile(sample_b, 100 * (alpha / 2.))
	print('ARAP mean, stdev, low, high: ', mean_b, std_b, low_b, high_b)
	print('ARAP mean, conf interval: ', bs.bootstrap(value_b, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=value_b.size, num_iterations=n_bootstrap_samples))

	sample_c = np.array(sample_c)
	mean_c = np.mean(sample_c)
	std_c = np.std(sample_c)
	low_c = 2 * mean_c - np.percentile(sample_c, 100 * (1 - alpha / 2.))
	high_c = 2 * mean_c - np.percentile(sample_c, 100 * (alpha / 2.))
	print('DragDiffusion mean, stdev, low, high: ', mean_c, std_c, low_c, high_c)
	print('DragDiffusion mean, conf interval: ', bs.bootstrap(value_c, stat_func=bs_stats.mean, alpha=alpha, iteration_batch_size=value_c.size, num_iterations=n_bootstrap_samples))


	df.to_csv('upload.csv')
	exit()
    

	return mean, stdev, low, high, num_workers_passed, samples


'''
Analyze a whole directory of batch result .csv's and write the results to another
   .csv (so we can put that into Tableau/whatever)
'''
def analyze_to_csv(in_dir, out_filename):
	outfile = open(out_filename, 'w')
	header = 'experiment_name,conditionA,conditionB,conditionC,mean,stdev,ci_low,ci_high,num_workers,num_samples\n'
	outfile.write(header)

	files = [f for f in os.listdir(in_dir) if f.endswith('.csv')]

	total_num_workers = 0
	total_num_samples = 0
	all_samples = []
	for fname in files:
		print(fname)
		# m = re.search('(.*)_(.*)_vs_(.*).csv', fname)
		experiment_name = 'userstudy_apap_v1'
		conditionA = 'Ours'
		conditionB = 'ARAP'
		conditionC = 'DragDiffusion'
		fpath = os.path.join(in_dir, fname)
		mean, stdev, low, high, num_workers_passed, samples = analyze(fpath)
		all_samples.append(samples)
		total_num_workers += num_workers_passed
		num_samples = num_workers_passed * (n_comparisons - n_vigilance)
		total_num_samples += num_samples
		outfile.write(','.join([experiment_name, conditionA, conditionB, conditionC, str(mean), str(stdev), str(low), str(high), str(num_workers_passed), str(num_samples) , '\n']))

	all_samples = np.concatenate(all_samples,0)
	total_mean = np.mean(all_samples)
	total_std = np.std(all_samples)

	total_low = 2 * total_mean - np.percentile(all_samples, 100 * (1 - alpha / 2.))
	total_high = 2 * total_mean - np.percentile(all_samples, 100 * (alpha / 2.))

	outfile.write(','.join(['Total', conditionA, conditionB, conditionC, str(total_mean), str(total_std), str(total_low), str(total_high), str(total_num_workers), str(total_num_samples) , '\n']))
	outfile.close()

	print("total mean,  std", total_mean, total_std)

	with open(os.path.join(out_dir, "final_stats.txt"), "w") as f:
		f.write(f"mean:{total_mean}\nstd:{total_std}")

	print(f"Total number of workers: {total_num_workers}")


analyze_to_csv(in_dir, out_filename)