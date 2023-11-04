import argparse
import csv
import numpy as np
import os
from random import shuffle
from random import randint
import hashlib
import time


parser = argparse.ArgumentParser(description='Generate input CSV for MTurk 2AFC study')

# In a single HIT, how many comparisons will the Turker make?
parser.add_argument('--n_real_comparisons_per_hit', type=int, default=20)
# How many "vigilance tests" (i.e. 'obvious' comparisons to check if the Turker is paying attention) per HIT?
parser.add_argument('--n_vigilance_tests_per_hit', type=int, default=5)
# How many HITs?
parser.add_argument('--n_hits', type=int, default=10)
# Unique ID so that Turkers can only work on one HIT for any given experiment
# See http://uniqueturker.myleott.com/ for more information about how this works
parser.add_argument('--unique_id', type=str, default='')
# The name of the experiment
parser.add_argument('--experiment_name', type=str, default='20231103_2d_1')
# The name of condition O (Original image before editing)
parser.add_argument('--conditionO', type=str, default='Source')
# The name of condition A
parser.add_argument('--conditionA', type=str, default='Ours')
# The name of condition B
parser.add_argument('--conditionB', type=str, default='DragDiffusion')
# The name of condition C
parser.add_argument('--conditionC', type=str, default='ARAP')


args = parser.parse_args()
np.random.seed(2024)


n_real_comparisons_per_hit = args.n_real_comparisons_per_hit
n_vigilance_tests_per_hit = args.n_vigilance_tests_per_hit
n_comparisons_per_hit = n_real_comparisons_per_hit + n_vigilance_tests_per_hit
n_hits = args.n_hits
experiment_name = args.experiment_name
# unique_id = args.unique_id
unique_id = hashlib.md5(('APAP'+str(time.time())).encode('utf-8')).hexdigest()
conditionO = args.conditionO
conditionA = args.conditionA
conditionB = args.conditionB
conditionC = args.conditionC
# Compute the name of the output csv file
output_loc = f'data/{experiment_name}/{experiment_name}.csv'


#########################################################################################


f = open(output_loc,'w')

first_row = ''

first_row += 'unique_id,'

for comp_idx in range(n_comparisons_per_hit):
	first_row += 'gt_side' + str(comp_idx+1) + ','
	first_row += 'images_original' + str(comp_idx+1) + ','
	first_row += 'images_left' + str(comp_idx+1) + ','
	first_row += 'images_mid' + str(comp_idx+1) + ','
	first_row += 'images_right' + str(comp_idx+1) + ','	

first_row = first_row[:-1]
first_row += '\n'

f.write(first_row)


conditionO_dir = '{}/{}'.format(experiment_name, conditionO)
conditionA_dir = '{}/{}'.format(experiment_name, conditionA)
conditionB_dir = '{}/{}'.format(experiment_name, conditionB)
conditionC_dir = '{}/{}'.format(experiment_name, conditionC)
vigilance_source_dir = '{}/vigilance_source'.format(experiment_name)
vigilance_true_dir = '{}/vigilance_true'.format(experiment_name)
vigilance_random_dir = '{}/vigilance_random'.format(experiment_name)

for hit_idx in range(n_hits):
	row = ''
	row += unique_id + ','
	# Build the list of images for each condition
	images_o = ['{}/{}'.format(conditionO_dir, i) for i in range(1, n_real_comparisons_per_hit+1)]
	images_a = ['{}/{}'.format(conditionA_dir, i) for i in range(1, n_real_comparisons_per_hit+1)]
	images_b = ['{}/{}'.format(conditionB_dir, i) for i in range(1, n_real_comparisons_per_hit+1)]
	images_c = ['{}/{}'.format(conditionC_dir, i) for i in range(1, n_real_comparisons_per_hit+1)]
	
	# # Randomize order (this randomizes the pairing)
	# shuffle(images_a)
	# shuffle(images_b)
	# Randomize order, but keep pairs intact (so we compare against different conditions of the same item (e.g. mesh, scene, etc.))
	indices = list(range(0, len(images_a)))
	shuffle(indices)
	images_o = [images_o[i] for i in indices]
	images_a = [images_a[i] for i in indices]
	images_b = [images_b[i] for i in indices]
	images_c = [images_c[i] for i in indices]
	# Pick random indices at which to insert vigilance tests
	# (Insert the true image into images_a, and the false/random image into images_b)
	for v in range(0, n_vigilance_tests_per_hit):
		insert_idx = randint(0, len(images_a)-1)
		vigilance_source_img = '{}/{}'.format(vigilance_source_dir, v+1)
		images_o.insert(insert_idx, vigilance_source_img)
		vigilance_true_img = '{}/{}'.format(vigilance_true_dir, v+1)
		images_a.insert(insert_idx, vigilance_true_img)
		vigilance_random_img = '{}/{}'.format(vigilance_random_dir, v+1)
		images_b.insert(insert_idx, vigilance_random_img)
		images_c.insert(insert_idx, vigilance_random_img)
	# Add columns for each comparison
	for comp_idx in range(n_comparisons_per_hit):
		image_o = images_o[comp_idx]
		image_a = images_a[comp_idx]
		image_b = images_b[comp_idx]
		image_c = images_c[comp_idx]
		# Choose whether to put A on the left or on the middle or on the right
		a_loc = np.random.choice([0,1,2])
		if a_loc == 0:
			# Add gt_side
			row += 'a,'
			# Add images_original
			row += image_o + ','
			# Add images_left
			row += image_a + ','
			# Add images_mid
			row += image_b + ','
			# Add images_right
			row += image_c + ','
		elif a_loc == 1:
			# Add gt_side
			row += 'b,'
			# Add images_original
			row += image_o + ','
			# Add images_left
			row += image_c + ','
			# Add images_mid
			row += image_a + ','
			# Add images_right
			row += image_b + ','
		else:
			# Add gt_side
			row += 'c,'
			# Add images_original
			row += image_o + ','
			# Add images_left
			row += image_b + ','
			# Add images_mid
			row += image_c + ','
			# Add images_right
			row += image_a + ','
	# Write row to file
	row = row[:-1]
	row += '\n'
	f.write(row)

f.close()