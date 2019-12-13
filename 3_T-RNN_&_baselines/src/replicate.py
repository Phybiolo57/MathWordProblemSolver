######################################################################
# File: replicate.py
# Author: Vishal Dey
# Created on: 11 Dec 2019
#######################################################################
'''
 Synopsis: Duplicates Dolphin300 to Dolphin1500 from Dolphin18k
	duplicates templates, substitutes quantities and recompute ans
'''

import os
import sys
import json
import numpy as np
from copy import deepcopy

from utils import *

np.random.seed(123)

def read_json(fname):
	with open(os.path.join('./data/', fname), 'r') as fp:
		return json.load(fp)

def write_json(fname, data):
	with open(os.path.join('./data/', fname), 'w') as fp:
		json.dump(data, fp)


def duplicate(entry):
	text = entry['text']
	#equation = entry['expression']
	num_list = entry['num_list']
	post_temp = entry['post_template']

	dup_num_list = []
	dup_text = []

	for i in num_list:
		if '.' in i:
			temp = str(np.random.rand())[:len(i)]
			while (temp in num_list or temp in dup_num_list):
				temp = str(np.random.rand())[:len(i)]
			dup_num_list.append(temp)
		else:
			temp = str(np.random.randint(123))
			while (temp in num_list or temp in dup_num_list):
				temp = str(np.random.randint(10000))
			dup_num_list.append(temp)

	for each in text.split():
		if each in num_list:
			dup_text.append(dup_num_list[num_list.index(each)])
		else:
			dup_text.append(each)

	post_equ = []
	for i in post_temp:
		if 'temp' in i:
			num = dup_num_list[int(i.split('_')[1])]
			post_equ.append(num)
		else:
			post_equ.append(i)
	try:
		ans = post_solver(post_equ)
	except:
		return None
#	print(dup_num_list, ' '.join(dup_text), ans)
	return dup_num_list, ' '.join(dup_text), ans


def main():
	dolphin = read_json('final_dolphin_data.json')

	n_repeats = 5
	max_id = int(max(list(dolphin.keys())))

	new_dolphin = {}

	for k, v in dolphin.items():
		new_dolphin[k] = deepcopy(v)
	
		for i in range(n_repeats):
			max_id += 1
			tmp = duplicate(v)
			if tmp:
				new_dolphin[str(max_id)] = deepcopy(v)
				new_dolphin[str(max_id)]['index'] = str(max_id)
				new_dolphin[str(max_id)]['text'] = tmp[1]
				new_dolphin[str(max_id)]['num_list'] = tmp[0]
				new_dolphin[str(max_id)]['ans'] = tmp[-1]
#		print(new_dolphin)

	print(len(new_dolphin.keys()))
	
	write_json('final_dolphin_data_replicate.json', new_dolphin)
	index = list(new_dolphin.keys())

	valid_size = int(0.2*len(index))
	
	valid_ids = np.random.choice(index, valid_size)
	test_size = int(valid_size/2)
	valid_size -= test_size

	print(len(index)-valid_size-test_size, valid_size, test_size)

	# validation ids
	write_json('valid_ids_dolphin.json', valid_ids[:valid_size].tolist())
	# test ids
	fp1 = open('./data/id_ans_test_dolphin', 'w')

	testtemp = []
	for id in valid_ids[valid_size:]:
		print(id + '\t' + new_dolphin[id]['ans'], file=fp1)
		tmp = ' '.join(new_dolphin[id]['post_template'])
		for ch in ['+', '-', '*', '/']:
			tmp = tmp.replace(ch, '^')
		testtemp.append([id, tmp.split()])
	
	fp1.close()
	
	# test templates masked
	write_json('pg_norm_test_dolphin.json', testtemp)
	

main()
