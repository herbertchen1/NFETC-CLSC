from model import Model
import tensorflow as tf
from utils import data_utils, prior_utils
from utils import eval_utils
import numpy as np
import config
import pickle
from functools import reduce


tf.set_random_seed(seed=config.RANDOM_SEED)


class NFETC(Model):
	def __init__(self, sequence_length, mention_length, num_classes, vocab_size,
	             embedding_size, position_size, pretrained_embedding, wpe, type_info, hparams):
		self.sequence_length = sequence_length
		self.mention_length = mention_length
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.position_size = position_size
		self.pretrained_embedding = pretrained_embedding
		self.wpe = wpe

		self.state_size = hparams.state_size
		self.hidden_layers = hparams.hidden_layers
		self.hidden_size = hparams.hidden_size
		self.wpe_dim = hparams.wpe_dim
		self.l2_reg_lambda = hparams.l2_reg_lambda
		self.lr = hparams.lr

		self.dense_keep_prob = hparams.dense_keep_prob
		self.rnn_keep_prob = hparams.rnn_keep_prob

		self.rnn_dense_dropoutkeeper=self.dense_keep_prob
		self.hp=hparams
		self.batch_size = hparams.batch_size
		self.num_epochs = hparams.num_epochs
		#self.bn=hparams.bn

		self.prior = tf.Variable(prior_utils.create_prior(type_info), trainable=False, dtype=tf.float32,
		                         name="prior")  # all one;no alpha
		self.alpha=hparams.alpha

		self.useCCLPloss=hparams.useCCLPloss

		self.makchainTimeForlabel=hparams.makchainlabel#directly use prob not propagate
		self.makchainTimeForfeature=hparams.makchainfeature

		#self.sslloss=hparams.sslloss
		self.measureway=hparams.measureway

		#self.filterdata=hparams.filterdata

		self.tune = tf.Variable(np.transpose(prior_utils.create_prior(type_info, hparams.alpha)), trainable=False,
		                        dtype=tf.float32, name="tune")  # tr 之后，每个节点算的就是整个path的penelty 概率

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.build()

	def add_placeholders(self):
		self.input_words = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_words")
		self.input_textlen = tf.placeholder(tf.int32, [None], name="input_textlen")
		self.input_mentions = tf.placeholder(tf.int32, [None, self.mention_length], name="input_mentions")
		self.input_mentionlen = tf.placeholder(tf.int32, [None], name="input_mentionlen")
		self.input_positions = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_positions")
		self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes], name="input_labels")
		self.cclpvar = tf.placeholder(tf.float32, name="cclpvar")

		self.phase = tf.placeholder(tf.bool, name="phase")

		self.dense_dropout = tf.placeholder(tf.float32, name="dense_dropout")

		self.rnn_dropout = tf.placeholder(tf.float32, name="rnn_dropout")
		self.rnn_dense_dropout = tf.placeholder(tf.float32, name="rnn_dense_dropout")
		self.words = tf.placeholder(tf.float32, name="words")
		self.movingmax=tf.Variable(0.0,trainable=False)
		tmp = [i for i in range(self.mention_length)]
		tmp[0] = self.mention_length
		interval = tf.Variable(tmp, trainable=False)
		interval_row = tf.expand_dims(interval, 0)
		upper = tf.expand_dims(self.input_mentionlen - 1, 1)
		mask = tf.less(interval_row, upper)
		self.mention = tf.where(mask, self.input_mentions, tf.zeros_like(self.input_mentions))
		self.mentionlen = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
		self.mentionlen = tf.cast(
			tf.where(tf.not_equal(self.mentionlen, tf.zeros_like(self.mentionlen)), self.mentionlen,
			         tf.ones_like(self.mentionlen)), tf.float32)
		self.mentionlen = tf.expand_dims(self.mentionlen, 1)

	def create_feed_dict(self, input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
	                     input_labels=None, phase=False, dense_dropout=1., rnn_dropout=1.,rnn_dense_dropout=1.,cclpvar=None):
		#print(rnn_dense_dropout)
		feed_dict = {
			self.input_words: input_words,
			self.input_textlen: input_textlen,
			self.input_mentions: input_mentions,
			self.input_mentionlen: input_mentionlen,
			self.input_positions: input_positions,
			self.phase: phase,
			self.dense_dropout: dense_dropout,
			self.rnn_dropout: rnn_dropout,
			self.rnn_dense_dropout:rnn_dense_dropout,
		}
		feed_dict[self.cclpvar] =0
		if input_labels is not None:
			feed_dict[self.input_labels] = input_labels
		if cclpvar is not None:
			feed_dict[self.cclpvar] = cclpvar
		return feed_dict


	def add_embedding(self):
		with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
			W = tf.Variable(self.pretrained_embedding, trainable=False, dtype=tf.float32, name="W")
			self.embedded_words = tf.nn.embedding_lookup(W, self.input_words)
			self.embedded_mentions = tf.nn.embedding_lookup(W, self.input_mentions)
			self.mention_embedding = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(W, self.mention),
			                                                 axis=1),self.mentionlen)

		with tf.device('/cpu:0'), tf.name_scope("position_embedding"):
			W = tf.Variable(self.wpe, trainable=False, dtype=tf.float32, name="W")
			self.wpe_chars = tf.nn.embedding_lookup(W, self.input_positions)
		self.input_sentences = tf.concat([self.embedded_words, self.wpe_chars], 2)


	def add_hidden_layer(self, x, idx):
		dim = self.feature_dim if idx == 0 else self.hidden_size
		with tf.variable_scope("hidden_%d" % idx):
			W = tf.get_variable("W", shape=[dim, self.hidden_size],
			                    initializer=tf.contrib.layers.variance_scaling_initializer(
				                    seed=config.RANDOM_SEED,factor=2.0))
			b = tf.get_variable("b", shape=[self.hidden_size],
			                    initializer=tf.contrib.layers.variance_scaling_initializer(
				                    seed=config.RANDOM_SEED,factor=2.0))
			h = tf.nn.xw_plus_b(x, W, b)
			h_drop = tf.nn.dropout(tf.nn.relu(h), self.dense_dropout, seed=config.RANDOM_SEED)
		return h_drop

	def extract_last_relevant(self, outputs, seq_len):
		batch_size = tf.shape(outputs)[0]
		max_length = int(outputs.get_shape()[1])
		num_units = int(outputs.get_shape()[2])
		index = tf.range(0, batch_size) * max_length + (seq_len - 1)
		flat = tf.reshape(outputs, [-1, num_units])
		relevant = tf.gather(flat, index)
		return relevant

	def add_prediction_op(self):
		self.add_embedding()
		self.lossesmask = 1 - self.boolize(tf.reduce_sum(self.input_labels, axis=1, keepdims=True, name='lmsk'),
		                                   threhold=1.1)

		self.numclean = tf.clip_by_value(tf.cast(
										 tf.reduce_sum(self.lossesmask), dtype=tf.float32),
		                                 clip_value_min=1e-10, clip_value_max=1000000)#一个batch 里面clean data的数量
		self.allonemasknoisy = tf.tile(self.lossesmask, multiples=[1, self.num_classes])# clean data 为 1* num_classes 的tensor
		self.lossesmaskMatrix = self.allonemasknoisy * self.input_labels

		self.bsize = tf.shape(self.embedded_mentions)[0]
		with tf.name_scope("sentence_repr"):
			attention_w = tf.get_variable("attention_w", [self.state_size, 1],
			                              initializer=tf.contrib.layers.variance_scaling_initializer(seed=config.RANDOM_SEED,factor=2.0))
			cell_forward = tf.contrib.rnn.LSTMCell(self.state_size)
			cell_backward = tf.contrib.rnn.LSTMCell(self.state_size)
			cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, input_keep_prob=self.rnn_dense_dropout,
			                                             output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)
			cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, input_keep_prob=self.rnn_dense_dropout,
			                                              output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

			outputs, states = tf.nn.bidirectional_dynamic_rnn(
				cell_forward, cell_backward, self.input_sentences,
				sequence_length=self.input_textlen, dtype=tf.float32)
			outputs_added = tf.nn.relu(tf.add(outputs[0], outputs[1]))
			alpha = tf.nn.softmax(tf.reshape(tf.matmul(
				tf.reshape(outputs_added, [-1, self.state_size]),
				attention_w),
				[-1, self.sequence_length]))
			alpha = tf.expand_dims(alpha, 1)
			self.sen_repr = tf.reshape(tf.squeeze(tf.matmul(alpha, outputs_added)),[self.bsize,self.state_size])

		with tf.name_scope("mention_repr"):
			cell = tf.contrib.rnn.LSTMCell(self.state_size)
			cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.rnn_dense_dropout,
				                                     output_keep_prob=self.rnn_dropout, seed=config.RANDOM_SEED)

			outputs, states = tf.nn.dynamic_rnn(
					cell, self.embedded_mentions,
					sequence_length=self.input_mentionlen, dtype=tf.float32)
			self.men_repr = self.extract_last_relevant(outputs, self.input_mentionlen)

			self.features = tf.concat([self.sen_repr,self.men_repr,self.mention_embedding], -1)
			self.feature_dim = self.state_size * 2 + self.embedding_size

		h_bn = self.features
		hcclp = tf.nn.relu(h_bn)
		h_output = tf.nn.dropout(hcclp, self.dense_dropout, seed=config.RANDOM_SEED)

		#get representation layer
		for i in range(self.hidden_layers):
			h_output = self.add_hidden_layer(h_output, i)
		if self.hidden_layers == 0:
			self.hidden_size = self.feature_dim

		with tf.variable_scope("typeVec",reuse=tf.AUTO_REUSE):
			W = tf.get_variable("W", shape=[ self.hidden_size,self.num_classes],
			                          initializer=tf.contrib.layers.variance_scaling_initializer(seed=config.RANDOM_SEED,factor=2.0))  # hidden size= 660
			b = tf.get_variable("b", shape=[self.num_classes],
			                    initializer=tf.contrib.layers.variance_scaling_initializer(seed=config.RANDOM_SEED,factor=2.0))

			self.scores = tf.nn.xw_plus_b(h_output, W, b, name="scores")  # [batch,num class]

			#将input label的父节点都找到
			self.labelpth=self.boolize(tf.matmul(self.input_labels, self.prior), threhold=0.0)

			#不加hier的概率
			self.proba =tf.clip_by_value( tf.nn.softmax(self.scores, axis=1),1e-10,1)
			#加了hier的概率
			self.adjusted_proba = tf.matmul(self.proba, self.tune)
			self.adjusted_proba = tf.clip_by_value(self.adjusted_proba, 1e-10, 1, name='adprob')

			# unleaked ori props
			self.maxtype = tf.argmax(self.proba, 1, name="maxtype")
			self.predictions = tf.one_hot(self.maxtype, self.num_classes, name='prediction')

			p = self.distanceMeasure(h_output, measureway=self.measureway)

			if self.useCCLPloss:
				choicematrix = self.input_labels
				self.LP_post,self.cclploss=self.calculateCClP(p,choicematrix,featurestep=self.makchainTimeForfeature)
				self.fi=tf.stop_gradient(self.LP_post)

	def calculateCClP(self,p,choicematrix,featurestep):
		H = tf.nn.softmax(p, axis=1, name='transMat')  # vote with himself

		# random init
		fi = tf.stop_gradient(
			tf.random_uniform(name='fi',
			                  shape=[self.bsize, self.num_classes],
			                  minval=0.0001, maxval=1.0,
			                  seed=config.RANDOM_SEED))
		fi = tf.div(fi , tf.tile(tf.reduce_sum(fi, axis=-1, keepdims=True),
		                         multiples=[1, self.num_classes]),name='oriP')

		# loop exit function
		cond = lambda fi_, distance, i: i < self.makchainTimeForlabel

		# loop body of LP
		def body(fi_, distance, i):
			fi_ = tf.matmul(H, fi_) * choicematrix
			fi_ = tf.div(fi_,
			            tf.tile(tf.reduce_sum(fi_, axis=-1, keepdims=True),
			                    multiples=[1, self.num_classes]))
			i += 1
			return (fi_, distance, i)

		# loop of LP
		fi, _, _ = tf.while_loop(cond=cond, body=body, loop_vars=(fi, H, 0.0))

		# calculate the mass of each types
		m = tf.reduce_sum(fi, axis=0, keep_dims=True,name='maskM')
		fij = tf.div(fi, tf.tile(tf.clip_by_value(m, 1e-10, 1), multiples=[tf.shape(H)[0], 1]))

		# T is the desirable transition matrix
		T = tf.matmul(fi, tf.transpose(fij), name='desTransitionMatrix')

		# Transition masking matrix
		M = tf.matmul(fi, tf.transpose(fi))

		cclpLoss = 0.0

		# L clsc
		Hs = H
		# loop exit
		cclpcond = lambda H, M, Hs, i, cclpLoss,noisymask,numclean: i < featurestep

		i = tf.Variable(1.0, trainable=False, dtype=tf.float32)

		# loop body
		def cclpbody(H, M, Hs, i, cclpLoss,noisymask,numclean):
			clscKLmat=T * tf.log(tf.clip_by_value(Hs, 1e-10, 1))
			m = tf.reduce_mean(tf.reduce_mean(clscKLmat))
			Hs = tf.matmul((H * M), Hs)
			cclpLoss -= tf.div(m, i)
			i = i + 1
			return (H, M, Hs, i, cclpLoss,noisymask,numclean)

		H, M, Hs, i, cclpLoss,_,_ = tf.while_loop(cond=cclpcond, body=cclpbody,
		                                      loop_vars=(H, M, Hs, i, cclpLoss,
		                                                 tf.tile(self.lossesmask, multiples=[1, self.bsize]),
		                                                 self.numclean))
		# return LP pro and cclploss
		return fi,cclpLoss


	def distanceMeasure(self,h_output,measureway='dot-product'):
		'''
			距离矩阵计算方式，目前定义了 dot-product，scale dot-product，cos
		'''
		if measureway=='dot-product':
			hp=h_output
			distance=tf.matmul(hp, tf.transpose(h_output)) / tf.sqrt(tf.cast(tf.shape(h_output)[1],dtype=tf.float32))

		elif measureway=='dot-product-noscale':
			distance=tf.matmul(h_output, tf.transpose(h_output))
		elif measureway=='cosine':
			mod=tf.sqrt(tf.reduce_sum(h_output * h_output, axis=1,keep_dims=True))
			hihj=tf.matmul(h_output, tf.transpose(h_output))
			hi2hj2=tf.tile(mod,multiples=[1,tf.shape(mod)[0]])*tf.tile(tf.transpose(mod),multiples=[tf.shape(mod)[0],1])
			coshij=tf.div(hihj,hi2hj2)
			distance=tf.nn.softmax(-coshij,axis=-1)
		else:
			assert False,'you must define distance function'

		return distance

	#用于bool化矩阵
	def boolize(self, item, threhold):
		return tf.cast(tf.greater(item, threhold), dtype=tf.float32)

	def add_loss_op(self):

		with tf.name_scope("loss"):
			self.comp = tf.Variable(0.0)

			proba=self.adjusted_proba

			#clean data  loss function
			numclean=self.numclean

			losses = -tf.reduce_sum(tf.reduce_sum(tf.multiply(self.lossesmaskMatrix,
					                                tf.log(tf.clip_by_value(proba, 1e-10, 1),
					                                name='labelagreeprob'),name='ssloss'),axis=1))\
					                                /numclean

			self.suploss = losses


			if self.useCCLPloss:
				print('use cclploss')
				losses += self.cclpvar*self.cclploss  # use cclp loss

			self.l2_loss = tf.contrib.layers.apply_regularization(
				regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
				weights_list=tf.trainable_variables())
			self.loss = losses + self.l2_loss

	def add_training_op(self):

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.grads_and_vars = optimizer.compute_gradients(self.loss)

		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(extra_update_ops):
			self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

	def train_on_batch(self, sess, input_words, input_textlen, input_mentions,
	                   input_mentionlen, input_positions,input_labels,cclpvar=None):
		feed = self.create_feed_dict(input_words, input_textlen, input_mentions, input_mentionlen, input_positions,
		                             input_labels, True, self.dense_keep_prob, self.rnn_keep_prob,
		                             rnn_dense_dropout=self.rnn_dense_dropoutkeeper,
		                             cclpvar=cclpvar)
		Variablelist=[self.train_op, self.global_step, self.loss,self.l2_loss,self.suploss]
		Variablename=['_','step','loss','L2_Loss','CCE_loss']

		if self.useCCLPloss:
			Variablelist.append(self.cclploss)
			Variablename.append('Clsc_Loss')
		a= sess.run(
				Variablelist,
				feed_dict=feed)
		step=a[1]
		if step:
			outpair=list(zip(Variablename,a))
			outstring=''
			for k,v in outpair:
				if k=='_':
					continue
				outstring+=k+': '+str(v)[:8]+' '
			print(outstring)

	def get_scores(self, preds, labels,id2type):
		label_path=eval_utils.label_path
		if type(preds) == np.ndarray:
			preds = [[label_path(id2type[i]) for i, x in enumerate(line) if x > 0] for line in preds]
			preds = [list(set(reduce(lambda x, y: x + y, line))) for line in preds]
		else:
			preds = [label_path(id2type[x]) for x in preds]

		def vec2type(v):
			s = []
			for i in range(len(v)):
				if v[i]:
					s.extend(label_path(id2type[i]))
			return set(s)
		labels_test = [vec2type(x) for x in labels]  # path will caculate the father node for strict acc
		acc = eval_utils.strict(labels_test, preds)
		_, _, macro = eval_utils.loose_macro(labels_test, preds)
		_, _, micro = eval_utils.loose_micro(labels_test, preds)

		return acc, macro, micro

	def predict(self, sess, test):
		batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
		all_predictions = []
		all_labels = []
		all_maxtype = []
		for batch in batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch= zip(*batch)

			feed = self.create_feed_dict(words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch)
			batch_predictions, batchmaxtype = sess.run([self.predictions, self.maxtype], feed_dict=feed)
			if len(all_predictions) == 0:
				all_predictions = batch_predictions
			else:
				all_predictions = np.concatenate([all_predictions, batch_predictions])
			if len(all_maxtype) == 0:
				all_maxtype = batchmaxtype
			else:
				all_maxtype = np.concatenate([all_maxtype, batchmaxtype])

			if len(all_labels) == 0:
				all_labels = np.array(labels_batch)
			else:
				all_labels = np.concatenate([all_labels, np.array(labels_batch)])
		return all_predictions, all_maxtype

	def evaluate(self, sess, train, test):
		print('begin training')
		train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)

		data_size = len(train)
		num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
		epoch=0
		for batch in train_batches:
			words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch, labels_batch= zip(*batch)
			self.train_on_batch(sess, words_batch, textlen_batch, mentions_batch, mentionlen_batch, positions_batch,
			                    labels_batch, cclpvar=self.hp.cclpvar)
			current_step = tf.train.global_step(sess, self.global_step)
			if current_step % num_batches_per_epoch == 0:
				epoch+=1
				yield self.predict(sess, test)

