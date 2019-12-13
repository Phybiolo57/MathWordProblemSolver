from utils import *
from model import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
	def __init__(self, data_loader, params):
		self.data_loader = data_loader
		self.params = params
		self.train_len = len(data_loader.train_list)
		self.valid_len = len(data_loader.valid_list)
		self.test_len = len(data_loader.test_list)
		#self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0.0)
		#self.pg_seq = dict(read_data_json("./data/pg_seq_norm_0828.json")) #dict
		self.pg_seq = dict(read_data_json("./data/pg_norm_test_dolphin.json")) #dict
	
	def _train_batch_recur(self, model, batch_encode_pad_idx, batch_encode_num_pos, batch_encode_len, batch_gd_tree):
		batch_encode_pad_idx_tensor = torch.LongTensor(batch_encode_pad_idx)
		batch_encode_tensor_len = torch.LongTensor(batch_encode_len)
		
		batch_encode_pad_idx_tensor = batch_encode_pad_idx_tensor.to(device)#cuda()
		batch_encode_tensor_len = batch_encode_tensor_len.to(device)#cuda()
		#print ("batch_encode_num_pos",batch_encode_num_pos)
		b_pred, b_loss, b_count, b_acc_e, b_acc_e_t, b_acc_i = model(batch_encode_pad_idx_tensor, batch_encode_tensor_len, \
			  batch_encode_num_pos, batch_gd_tree)
		self.optimizer.zero_grad()
		#print (b_loss)
		b_loss.backward(retain_graph=True)
		clip_grad_norm_(model.parameters(), 5, norm_type=2.)
		self.optimizer.step()
		return b_pred, b_loss.item(), b_count, b_acc_e, b_acc_e_t, b_acc_i

	def _test_recur(self, model, data_list):
		batch_size = self.params['batch_size']
		data_generator = self.data_loader.get_batch(data_list, batch_size)
		test_pred = []
		test_count = 0
		test_acc_e = []
		test_acc_e_t = []
		test_acc_i = []
		for batch_elem in data_generator:
			batch_encode_idx = batch_elem['batch_encode_idx']
			batch_encode_pad_idx = batch_elem['batch_encode_pad_idx']
			batch_encode_num_pos = batch_elem['batch_encode_num_pos']
			batch_encode_len = batch_elem['batch_encode_len']

			batch_decode_idx = batch_elem['batch_decode_idx']

			batch_gd_tree = batch_elem['batch_gd_tree']
			
			batch_encode_pad_idx_tensor = torch.LongTensor(batch_encode_pad_idx)
			batch_encode_tensor_len = torch.LongTensor(batch_encode_len)

			batch_encode_pad_idx_tensor = batch_encode_pad_idx_tensor.to(device)#cuda()
			batch_encode_tensor_len = batch_encode_tensor_len.to(device)#cuda()
			
			b_pred, b_count, b_acc_e, b_acc_e_t, b_acc_i = model.test_forward_recur(batch_encode_pad_idx_tensor, batch_encode_tensor_len, \
			  batch_encode_num_pos, batch_gd_tree)
			
			test_pred+= b_pred
			test_count += b_count
			test_acc_e += b_acc_e
			test_acc_e_t += b_acc_e_t
			test_acc_i += b_acc_i
		return test_pred, test_count, test_acc_e, test_acc_e_t, test_acc_i

	def predict_joint_batch(self, model, batch_encode_pad_idx, batch_encode_num_pos, batch_encode_len, batch_gd_tree, batch_index, batch_num_list):
		batch_encode_pad_idx_tensor = torch.LongTensor(batch_encode_pad_idx)
		batch_encode_tensor_len = torch.LongTensor(batch_encode_len)
		
		batch_encode_pad_idx_tensor = batch_encode_pad_idx_tensor.to(device)#cuda()
		batch_encode_tensor_len = batch_encode_tensor_len.to(device)#cuda()
		
		#alphas = 'abcdefghijklmnopqrstuvwxyz'
		alphas = list(map(str, list(range(0, 14))))
		batch_seq_tree = []
		batch_flags = []
		batch_num_len = []
		for i in range(len(batch_index)):
			index = batch_index[i]
			op_template = self.pg_seq[index]
			new_op_temps = []
			num_len = 0
			#print(op_template)
			for temp_elem in op_template:
				if 'temp' in temp_elem:
					num_idx = alphas.index(temp_elem[5:])
					#num_idx = temp_elem[5:]
					#new_op_temps.append('temp_'+str(num_idx))
					new_op_temps.append(temp_elem)
					if num_len < num_idx:
						num_len = num_idx
				else:
					new_op_temps.append(temp_elem)
			
			#print (new_op_temps)
			#print (op_template)
			#print 
			#print ("0000", op_template)
			try:
				temp_tree = construct_tree_opblank(new_op_temps[:])
			except:
			#print ("error:", new_op_temps)
				temp_tree = construct_tree_opblank(['temp_0', 'temp_1', '^'])
			batch_seq_tree.append(temp_tree)
			num_list = batch_num_list[i]
			if num_len >= len(num_list):
				print ('error num len',new_op_temps, num_list)
				batch_flags.append(0)
			else:
				batch_flags.append(1)
			
			#print(index, temp_tree.__str__())
			#print(new_op_temps, num_list)
		
		#print('Batch flags: ', batch_flags) 
		#print(batch_index[0], batch_seq_tree[0].__str__())
		batch_pred_tree_node, batch_pred_post_equ = model.predict_forward_recur(batch_encode_pad_idx_tensor, batch_encode_tensor_len,\
							  batch_encode_num_pos, batch_seq_tree, batch_flags)
		
		return batch_pred_tree_node, batch_pred_post_equ
		#model.xxx(batch_encode_pad_idx_tensor, batch_encode_tensor_len, batch_encode_num_pos, batch_gd_tree, batch_flags)
	
	def predict_joint(self, model):
		batch_size = self.params['batch_size']
		data_generator = self.data_loader.get_batch(self.data_loader.test_list, batch_size)
		test_pred = []
		test_count = 0
		test_acc_e = []
		test_acc_e_t = []
		test_acc_i = []
		
		test_temp_acc = 0.0
		test_ans_acc = 0.0
		
		save_info = []
		
		for batch_elem in data_generator:
			batch_encode_idx = batch_elem['batch_encode_idx'][:]
			batch_encode_pad_idx = batch_elem['batch_encode_pad_idx'][:]
			batch_encode_num_pos = batch_elem['batch_encode_num_pos'][:]
			batch_encode_len = batch_elem['batch_encode_len'][:]

			batch_decode_idx = batch_elem['batch_decode_idx'][:]

			batch_gd_tree = batch_elem['batch_gd_tree'][:]
			
			batch_index = batch_elem['batch_index']
			batch_num_list = batch_elem['batch_num_list'][:]
			batch_solution = batch_elem['batch_solution']
			batch_post_equation = batch_elem['batch_post_equation']
			
			#b_pred, b_count, b_acc_e, b_acc_e_t, b_acc_i = 
			batch_pred_tree_node, batch_pred_post_equ = self.predict_joint_batch(model, batch_encode_pad_idx, batch_encode_num_pos,batch_encode_len, batch_gd_tree, batch_index, batch_num_list)
			
			for i in range(len(batch_solution)):
				pred_post_equ = batch_pred_post_equ[i]
				#pred_ans = batch_pred_ans[i]
				gold_post_equ = batch_post_equation[i]
				gold_ans = batch_solution[i]
				idx = batch_index[i]
				pgseq = self.pg_seq[idx]
				num_list = batch_num_list[i]
			
				#print (pred_post_equ)
				#print (pgseq)
				#print (gold_post_equ)
				#print (num_list)
				
				if pred_post_equ == []:
					pred_ans = -float('inf')
				else:
					pred_post_equ_ali = []
					for elem in pred_post_equ:
						if 'temp' in elem:
							num_idx = int(elem[5:])
							num_marker = num_list[num_idx]
							pred_post_equ_ali.append(str(num_marker))
						elif 'PI' == elem:
							pred_post_equ_ali.append("3.141592653589793")
						else:
							pred_post_equ_ali.append(elem)
					try:
						pred_ans = post_solver(pred_post_equ_ali)
					except:
						pred_ans = -float('inf')
			
			
				if abs(float(pred_ans)-float(gold_ans)) < 1e-5:
					test_ans_acc += 1
				if ' '.join(pred_post_equ) == ' '.join(gold_post_equ):
					test_temp_acc += 1
				#print (pred_ans, gold_ans)
				#print ()
				save_info.append({"idx":idx, "pred_post_eq":pred_post_equ, "gold_post_equ":gold_post_equ, "pred_ans":pred_ans,"gold_ans": gold_ans})
			
			
			#test_pred+= b_pred
			#test_count += b_count
			#test_acc_e += b_acc_e
			#test_acc_e_t += b_acc_e_t
			#test_acc_i += b_acc_i
		print ("final test temp_acc:{}, ans_acc:{}".format(test_temp_acc/self.test_len, test_ans_acc/self.test_len))
		#logging.debug("final test temp_acc:{}, ans_acc:{}".format(test_ans_acc/self.test_len, test_ans_acc/self.test_len))
		write_data_json(save_info, "./result_recur/dolphin_rep/save_info_"+str(test_temp_acc/self.test_len)+"_"+str(test_ans_acc/self.test_len)+".json")
		return save_info

	
	def _train_recur_epoch(self, model, start_epoch, n_epoch):
		batch_size = self.params['batch_size']
		data_loader = self.data_loader
		train_list = data_loader.train_list 
		valid_list = data_loader.valid_list
		test_list = data_loader.test_list
		
		
		valid_max_acc = 0 
		for epoch in range(start_epoch, n_epoch + 1):
			epoch_loss = 0
			epoch_pred = []
			epoch_count = 0
			epoch_acc_e = []
			epoch_acc_e_t = []
			epoch_acc_i = []
			train_generator = self.data_loader.get_batch(train_list, batch_size)
			s_time = time.time()
			xx = 0
			for batch_elem in train_generator:
				batch_encode_idx = batch_elem['batch_encode_idx']
				batch_encode_pad_idx = batch_elem['batch_encode_pad_idx']
				batch_encode_num_pos = batch_elem['batch_encode_num_pos']
				batch_encode_len = batch_elem['batch_encode_len']
				
				batch_decode_idx = batch_elem['batch_decode_idx']
				
				batch_gd_tree = batch_elem['batch_gd_tree']
				
				
				b_pred, b_loss, b_count, b_acc_e, b_acc_e_t, b_acc_i = self._train_batch_recur(model, batch_encode_pad_idx, \
								batch_encode_num_pos,batch_encode_len, batch_gd_tree)
				epoch_loss += b_loss
				epoch_pred+= b_pred
				epoch_count += b_count
				epoch_acc_e += b_acc_e
				epoch_acc_e_t += b_acc_e_t
				epoch_acc_i += b_acc_i
				xx += 1
				#print (xx)
				#print (b_pred)
				#break
			
			
			e_time = time.time()
			
			#print ("ee": epoch_acc_e)
			#print ("et": epoch_acc_e_t)
			#print ("recur epoch: {}, loss: {}, acc_e: {}, acc_i: {} time: {}".\
			#		format(epoch, epoch_loss/epoch_count, sum(epoch_acc_e)*1.0/sum(epoch_acc_e_t), sum(epoch_acc_i)/epoch_count, \
			#		   (e_time-s_time)/60))

			valid_pred, valid_count, valid_acc_e, valid_acc_e_t, valid_acc_i = self._test_recur(model, valid_list)
			test_pred, test_count, test_acc_e, test_acc_e_t, test_acc_i = self._test_recur(model, test_list)

			#print ('**********1', test_pred) 
			#print ('**********2', test_count)
			#print ('**********3', test_acc_e)
			#print ('**********4', test_acc_e_t)
			#print ('**********5', test_acc_i)
			
			print ("RECUR EPOCH: {}, loss: {}, train_acc_e: {}, train_acc_i: {} time: {}".\
			   format(epoch, epoch_loss/epoch_count, sum(epoch_acc_e)*1.0/sum(epoch_acc_e_t), sum(epoch_acc_i)/epoch_count, \
				  (e_time-s_time)/60))
			
			print ("valid_acc_e: {}, valid_acc_i: {}, test_acc_e: {}, test_acc_i: {}".format(sum(valid_acc_e)*1.0/sum(valid_acc_e_t), sum(valid_acc_i)*1.0/valid_count, sum(test_acc_e)*1.0/sum(test_acc_e_t), sum(test_acc_i)*1.0/test_count))

			
			test_acc = sum(valid_acc_i)*1.0/valid_count
			if test_acc >= valid_max_acc:
				print ("originial", valid_max_acc)
				valid_max_acc = test_acc
				print ("saving...", valid_max_acc)
				if os.path.exists(self.params['save_file']):
					os.remove(self.params['save_file'])
				torch.save(model, self.params['save_file'])
				print ("saveing ok!")
				self.predict_joint(model)
			
				# save recursive data: format: {'id': , 'right_flag':, 'predict_result', 'ground_result'} 
				# save joint data ['id':, 'right_flag':, predict_result, ground_result]
			
			else:
				print ("jumping...")
	
			print ()
		
		
	def train(self, model, optimizer):
		self.optimizer = optimizer
		#self._train_recur_epoch(model, 0, 0)
		#self._train_pointer_epoch(model, 0, 0)
		#self._train_joint_epoch(model,0, 0)
		#self._train_recur_epoch(model, 1, 1)
		#self._train_pointer_epoch(model, 1, 1)
		#self._train_joint_epoch(model,1, 1)
		#self._train_joint_epoch(model,0, 10)
		self._train_recur_epoch(model, 0, 100)
