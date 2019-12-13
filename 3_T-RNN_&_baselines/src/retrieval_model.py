######################################################################
# File: retrieval_model.py
# Author: Vishal Dey
# Created on: 11 Dec 2019
#######################################################################
'''
 Synopsis: Create w2v for corresponding string description of each problem
	Reads in pretrianed Word2vec vectors and add each word vector to obtain
	phrase vectors.
	Either compute TF-IDF / w2v based cosine similarity
'''

import os
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from copy import deepcopy

# for reproducibility
np.random.choice(123)


# read json file
def read_json(fname):
	with open(os.path.join('./data/', fname), 'r') as fp:
		return json.load(fp)


# compute TF-iDF based most similar problem
def find_similar_tfidf(tfidf_matrix, tfidf_vector):
	cosine_similarities = linear_kernel(tfidf_vector, tfidf_matrix).flatten()
	related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
	#print(related_docs_indices[0], cosine_similarities[related_docs_indices[0]])
	return related_docs_indices[0]

# compute w2v cosine based most similar problem
def find_similar_w2v(w2vmatrix, w2vemb, query, EMB_DIM, minv=0, maxv=1):
	query_emb = []
	for word in query.split():
		if word in w2vemb:
			query_emb.append(np.array(list(map(float, w2vemb[word]))))
		else:
			query_emb.append(np.random.uniform(minv, maxv, EMB_DIM))
	
	cosine_similarities = linear_kernel(query_emb, w2vmatrix).flatten()
	related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
	print(related_docs_indices)
	return related_docs_indices[0]


# load w2v problem
def load_w2v(w2v_file):
	EMB_DIM = 0
	w2v_emb = {}
	minv = sys.float_info.max
	maxv = sys.float_info.min

	with open(os.path.join('./w2v', w2v_file), 'r') as fp:
		EMB_DIM = int(fp.readline().split()[1])
		for line in fp.readlines():
			tmp = line.split()
			tmp = list(map(float, tmp[1:]))
			minv = min(minv, min(tmp))
			maxv = max(maxv, max(tmp))

	print(EMB_DIM, minv, maxv)
	return w2v_emb, EMB_DIM, minv, maxv


# compute w2v matrix (problem x EMB_DIM)
def compute_w2vmatrix(docs, w2v_emb, EMB_DIM, minv=0, maxv=1):
	w2vmatrix = []
	for doc in docs:
		emb = [0]*EMB_DIM
		for word in doc.split():
			if word in w2v_emb:
				emb += np.array(list(map(float, w2v_emb[word])))
			else:
				emb += np.random.uniform(minv, maxv, EMB_DIM)
		emb /= len(doc.split())

		w2vmatrix.append(emb)
	return w2vmatrix


# post fix equation solver
def post_solver(post_equ):
	stack = [] 
	op_list = ['+', '-', '/', '*', '^']
	
#	if len(post_equ)-2*len([i for i in post_equ if (any(c.isdigit() for c in i))]) >= 0:
#		print(post_equ)
#		return 0
	for elem in post_equ:
		if elem not in op_list:
			op_v = elem
			if '%' in op_v:
				op_v = float(op_v[:-1])/100.0
			if op_v == 'PI':
				op_v = np.pi
			stack.append(str(op_v))
		elif elem in op_list:
			op_v_1 = stack.pop()
			op_v_1 = float(op_v_1)
			op_v_2 = stack.pop()
			op_v_2 = float(op_v_2)
			if elem == '+':
				stack.append(str(op_v_2+op_v_1))
			elif elem == '-':
				stack.append(str(op_v_2-op_v_1))
			elif elem == '*':
				stack.append(str(op_v_2*op_v_1))
			elif elem == '/':
				if op_v_1 == 0:
					return np.nan
				stack.append(str(op_v_2/op_v_1))
			else:
				stack.append(str(op_v_2**op_v_1))
	return stack.pop()

# get post fix equation from the template
def get_equation(template, num_list):
	equation = []
	for x in template:
		if 'temp' in x:
			equation.append(str(num_list[int(x.split('_')[1])]))
		else:
			equation.append(x)
	return equation



def main():
	math23k = read_json('math23k_final.json')
	#math23k = read_json('final_dolphin_data.json')

	# read in test IDs and answer -  gold truth
	test_ids = []
	with open('./data/id_ans_test', 'r') as fp:
		for line in fp.readlines():
			test_ids.append(line.split('\t')[0])
#	del math23k['1682']

	test_length = len(test_ids)

	train = {}
	test = {}
	corpus = []

	# make training corpus text
	for k, v in math23k.items():
		if k not in test_ids:
			train[k] = v
			corpus.append(v['template_text'])
		else:
			test[k] = v

	corpus = list(corpus)
	indices = list(train.keys())

	print(len(corpus), test_length)

	# tfidf vectorizer	
	vectorizer = TfidfVectorizer()
	corpus_tfidf = vectorizer.fit_transform(corpus)

	# load w2v
	#w2v_emb, EMB_DIM, minv, maxv = load_w2v('crawl-300d-2M.vec')
	#w2vmatrix = compute_w2vmatrix(corpus, w2v_emb, EMB_DIM, minv, maxv)

	#print(len(w2vmatrix))
	similar_indices = {}

	for k, v in test.items():
		query_tfidf = vectorizer.transform([v['template_text']])
		i = find_similar_tfidf(corpus_tfidf, query_tfidf)
		#i = find_similar_w2v(w2vmatrix, w2v_emb, v['template_text'], EMB_DIM, minv, maxv)
		#i = jaccard_similarity(corpus, v['template_text'])
		#print(i)
		similar_indices[k] = indices[i]

	num_correct = 0
	# outputs the prediction
	fout = open('pred.txt', 'w')

	for k, v in similar_indices.items():
		template = math23k[v]['post_template']
		num_list = math23k[k]['num_list']
		ans = math23k[k]['ans']

		if (max([int(s.split('temp_')[1]) for s in set(template) if 'temp' in s]) < len(num_list)):
			pred_ans = post_solver(get_equation(template, num_list))
			print(math23k[k]['index'], pred_ans, ans, file=fout)
			if (np.isclose(float(pred_ans), float(ans))):
				num_correct += 1
		else:
			print('wrong template prediction: mismatch with len of num list')

	print('Accuracy = ', str(float(num_correct)/test_length))
	fout.close()

main()
