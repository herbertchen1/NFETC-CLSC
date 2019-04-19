from model_param_space import ModelParamSpace
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from optparse import OptionParser
from utils import logging_utils, data_utils, embedding_utils, pkl_utils
from utils.eval_utils import strict, loose_macro, loose_micro, label_path, complete_path
import numpy as np
from sklearn.model_selection import ShuffleSplit
import os
import config
import datetime, pickle
import tensorflow as tf
from nfetc_clsc import NFETC

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

class Task:
	def __init__(self, model_name, data_name, cv_runs, params_dict, logger, portion=100,save_name=''):
		print("Loading data...")
		if portion<=100:# all the data, portion% clean + all noisy
			self.portion = '-'+str(portion) if portion != 100 else ''
		else:
			portion/=100# only clean data, portion% clean
			self.portion='-'+str(int(portion))+'-clean'
		print('run task on: ', self.portion,' dataset: ',data_name)
		if data_name == "ontonotes":
			words_train, mentions_train, positions_train, labels_train = data_utils.load(
				config.ONTONOTES_TRAIN_CLEAN+self.portion)
			words, mentions, positions, labels = data_utils.load(config.ONTONOTES_TEST_CLEAN)
			type2id, typeDict = pkl_utils._load(config.ONTONOTES_TYPE)
			num_types = len(type2id)
			type_info = config.ONTONOTES_TYPE
		elif data_name == "bbn":
			words_train, mentions_train, positions_train, labels_train = data_utils.load(
				config.BBN_TRAIN_CLEAN+self.portion)
			words, mentions, positions, labels = data_utils.load(config.BBN_TEST_CLEAN)
			type2id, typeDict = pkl_utils._load(config.BBN_TYPE)
			num_types = len(type2id)
			type_info = config.BBN_TYPE
		else:
			assert False,'you have to specify the name of dataset with -d (ie. bbn/....)'
		self.model_name = model_name
		self.savename = save_name
		self.data_name = data_name
		self.cv_runs = cv_runs
		self.params_dict = params_dict
		self.hparams = AttrDict(params_dict)
		#self.hparams.alpha=alpha
		self.logger = logger

		self.id2type = {type2id[x]: x for x in type2id.keys()}

		def type2vec(types):  # only terminal will be labeled
			tmp = np.zeros(num_types)
			for t in str(types).split():
				if t in type2id.keys():
					tmp[type2id[t]] = 1.0
			return tmp

		labels_train = np.array([type2vec(t) for t in labels_train])  # one hot vec'
		labels = np.array([type2vec(t) for t in labels])

		tempname=self.data_name+config.testemb
		tempname=os.path.join(config.PKL_DIR,tempname)
		if os.path.exists(tempname):
			self.embedding = pickle.load(open(tempname, 'rb'))
			print('embedding load over')
		else:
			self.embedding = embedding_utils.\
				Embedding.fromCorpus(config.EMBEDDING_DATA,list(words_train) + list(words),
			                         config.MAX_DOCUMENT_LENGTH, config.MENTION_SIZE)
			pickle.dump(self.embedding, open(tempname, 'wb'))
			print('embedding dump over')
		self.embedding.max_document_length=config.MAX_DOCUMENT_LENGTH

		print("Preprocessing data...")

		if True:
			textlen_train = np.array(
				[self.embedding.len_transform1(x) for x in words_train])  # with cut down len sequence
			words_train = np.array([self.embedding.text_transform1(x) for x in
			                        words_train])  # with cut down word id sequence and mask with zero <PAD>
			mentionlen_train = np.array([self.embedding.len_transform2(x) for x in mentions_train])  # mention len
			mentions_train = np.array(
				[self.embedding.text_transform2(x) for x in mentions_train])  # mention text indexer
			positions_train = np.array(
				[self.embedding.position_transform(x) for x in positions_train])  # start ,end position
			print('get train data')

			textlen = np.array([self.embedding.len_transform1(x) for x in words])
			words = np.array([self.embedding.text_transform1(x) for x in words])  # padding and cut down
			mentionlen = np.array([self.embedding.len_transform2(x) for x in mentions])
			mentions = np.array([self.embedding.text_transform2(x) for x in mentions])
			positions = np.array([self.embedding.position_transform(x) for x in positions])
			print('get test data')
			# pickle.dump([textlen_train, words_train, mentionlen_train, mentions_train, positions_train,
			#              textlen, words, mentionlen, mentions, positions
			#              ], open(os.path.join(self.data_name + config.prep+self.portion, 'wb'))
			# print('dump preprocessed data to pkl over...')
		# else:
			# textlen_train, words_train, mentionlen_train, mentions_train, \
			# positions_train, textlen, words, mentionlen, mentions, positions = pickle.load(
			# 	open(self.data_name + config.prep+self.portion, 'rb'))
			# print('load preprocessed data from pkl over...')

		#if True:
		ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
		for test_index, valid_index in ss.split(np.zeros(len(labels)), labels):  # 用index做划分
			textlen_test, textlen_valid = textlen[test_index], textlen[valid_index]
			words_test, words_valid = words[test_index], words[valid_index]
			mentionlen_test, mentionlen_valid = mentionlen[test_index], mentionlen[valid_index]
			mentions_test, mentions_valid = mentions[test_index], mentions[valid_index]
			positions_test, positions_valid = positions[test_index], positions[valid_index]
			labels_test, labels_valid = labels[test_index], labels[valid_index]

		self.train_set = list(
			zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train,))
		self.valid_set = list(
			zip(words_valid, textlen_valid, mentions_valid, mentionlen_valid, positions_valid, labels_valid,))
		self.test_set = list(zip(words_test, textlen_test, mentions_test, mentionlen_test, positions_test, labels_test,))

		self.full_test_set = list(zip(words, textlen, mentions, mentionlen, positions, labels,))

		self.labels_test = labels_test
		self.labels = labels
		self.labels_valid = labels_valid

		self.num_types = num_types
		self.type_info = type_info
		self.logger.info("train set size:%d, test set size: %d" % (len(self.train_set), len(self.full_test_set)))

		self.model = self._get_model()
		self.saver = tf.train.Saver(tf.global_variables())
		checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.checkpoint_prefix = os.path.join(checkpoint_dir, self.__str__())

	def __str__(self):
		return self.model_name + self.savename

	def _get_model(self):
		np.random.seed(config.RANDOM_SEED)
		#print(noisemask)
		kwargs = {
			"sequence_length": config.MAX_DOCUMENT_LENGTH,
			"mention_length": config.MENTION_SIZE,
			"num_classes": self.num_types,
			"vocab_size": self.embedding.vocab_size,
			"embedding_size": self.embedding.embedding_dim,
			"position_size": self.embedding.position_size,
			"pretrained_embedding": self.embedding.embedding,
			"wpe": np.random.random_sample((self.embedding.position_size, self.hparams.wpe_dim)),
			"type_info": self.type_info,
			"hparams": self.hparams,
		}
		if "nfetc" in self.model_name:
			return NFETC(**kwargs)
		else:
			raise AttributeError("Invalid model name!")

	def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
		for k, v in sorted(d.items()):
			if isinstance(v, dict):
				self.logger.info("%s%s:" % (prefix, k))
				self.print_param_dict(v, prefix + incr_prefix, incr_prefix)
			else:
				self.logger.info("%s%s: %s" % (prefix, k, v))

	def create_session(self):
		session_conf = tf.ConfigProto(
			intra_op_parallelism_threads=8,
			allow_soft_placement=True,
			log_device_placement=False)
		session_conf.gpu_options.allow_growth = True
		return tf.Session(config=session_conf)

	def get_scores(self, preds, target='fullset'):
		preds = [label_path(self.id2type[x]) for x in preds]
		#print(self.test_set[0])
		def vec2type(v):
			s = []
			for i in range(len(v)):
				if v[i]:
					s.extend(label_path(self.id2type[i]))
			return set(s)

		print('eval on ', target)
		if target == 'fullset':
			labels_test = [vec2type(x) for x in self.labels]  # path will caculate the father node for strict acc
		else:
			labels_test = [vec2type(x) for x in self.labels_valid]
		words=[self.embedding.i2w(k[0]) for k in self.full_test_set]
		mentions = [self.embedding.i2w(k[2]) for k in self.full_test_set]
		acc = strict(labels_test, preds,oridata=(words,mentions),modelname=self.savename)
		_, _, macro = loose_macro(labels_test, preds)
		_, _, micro = loose_micro(labels_test, preds)
		return acc, macro, micro

	def refit(self):
		self.logger.info("Params")
		self._print_param_dict(self.params_dict)
		self.logger.info("Evaluation for each epoch")
		self.logger.info("\t\tEpoch\t\tAcc\t\tMacro\t\tMicro")
		sess = self.create_session()

		print('retraining times: ', self.cv_runs)
		sess.run(tf.global_variables_initializer())

		maxbaseonvalid = ()

		vaacclist = []
		vamacrolist = []
		vamicrolist = []

		for i in range(self.cv_runs):
			if self.cv_runs > 1 and i !=0:
				print('reopen sess...')
				sess.close()
				sess = self.create_session()
				sess.run(tf.global_variables_initializer())
			maxvaacc = -1
			epochs = 0

			for preds, maxtype in self.model.evaluate(sess, self.train_set, self.full_test_set):
				epochs += 1
				acc, macro, micro = self.get_scores(maxtype)
				vapreds, _ = self.model.predict(sess, self.valid_set)
				vaacc, vamacro, vamicro = self.get_scores(_, target='vatestset')
				# vaacc=round(vaacc,3)
				# vamacro=round(vamacro,3)
				# vamicro=round(vamicro,3)
				cmp=vaacc
				if cmp >= maxvaacc:
					maxvaacc = cmp
					maxbaseonvalid = (epochs, acc, macro, micro, maxvaacc)
					self.logger.info(
						"\tep\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" % (epochs, vaacc, vamacro, vamicro, maxvaacc))
				else:
					self.logger.info("\tep\t%d\t\t%.3f\t\t%.3f\t\t%.3f" %
					                 (epochs, vaacc, vamacro, vamicro))

			vaacclist.append(maxbaseonvalid[1])
			vamacrolist.append(maxbaseonvalid[2])
			vamicrolist.append(maxbaseonvalid[3])
			self.logger.info("\tMax\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" % (
			maxbaseonvalid[0], maxbaseonvalid[1], maxbaseonvalid[2], maxbaseonvalid[3], maxbaseonvalid[4]))

		#计算 验证集最大时，测试集的方差与均值
		meanvaacc = np.mean(vaacclist)
		meanvamacro = np.mean(vamacrolist)
		meanvamicro = np.mean(vamicrolist)
		stdvaacc = np.std(vaacclist)
		stdvamacro = np.std(vamacrolist)
		stdvamicro = np.std(vamicrolist)

		self.logger.info("\tCV\t%.1f±%.1f\t%.1f±%.1f\t%.1f±%.1f"
		                 % (meanvaacc*100, stdvaacc*100, meanvamacro*100,
		                    stdvamacro*100, meanvamicro*100, stdvamicro*100))
		sess.close()

	def get_feature(self,dataset):
		if dataset=='train':
			ds=self.train_set
		elif dataset=='test':
			ds=self.full_test_set
		else:
			assert False,'you must give a test or training set '
		checkpoint_file = self.checkpoint_prefix
		print('begin  to reload model')
		sess = self.create_session()
		saver = tf.train.Saver()

		sess.run(tf.global_variables_initializer())
		saver.restore(sess,checkpoint_file)
		self.model.getfeatures(sess,ds)

	def save(self, sess):

		path = self.saver.save(sess, self.checkpoint_prefix)
		self.embedding.save(self.checkpoint_prefix)
		print("Saved model to {}".format(path))
		print('-' * 100)


class TaskOptimizer:
	def __init__(self, model_name, data_name, cv_runs, max_evals, logger, cvonfull, savename='',portion=100):
		self.model_name = model_name
		self.data_name = data_name
		self.cv_runs = cv_runs
		self.max_evals = max_evals
		self.logger = logger
		self.cvonfull = cvonfull
		self.save_name = savename
		self.model_param_space = ModelParamSpace(self.model_name)  # get the param dict via dict name
		self.portion=portion

	def _obj(self, param_dict):
		param_dict = self.model_param_space._convert_into_param(param_dict)
		self.task = Task(model_name=self.model_name,data_name= self.data_name,
		                 cv_runs=self.cv_runs, params_dict=param_dict, logger=self.logger, save_name=self.save_name,portion=self.portion)
		self.task.cv(self.cvonfull)
		tf.reset_default_graph()
		ret = {
			"loss": -self.task.eacc,
			# "attachments": {
			#     "pacc": self.task.pacc,
			#     # "eacc": self.task.eacc,
			# },
			"status": STATUS_OK
		}
		return ret

	def run(self):
		trials = Trials()
		best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
		best_params = space_eval(self.model_param_space._build_space(), best)
		best_params = self.model_param_space._convert_into_param(best_params)
		trial_loss = np.asarray(trials.losses(), dtype=float)
		best_ind = np.argmin(trial_loss)
		best_loss = -trial_loss[best_ind]
		# best_pacc = trials.trial_attachments(trials.trials[best_ind])["pacc"]
		# best_eacc = trials.trial_attachments(trials.trials[best_ind])["eacc"]
		self.logger.info("-" * 50)
		self.logger.info("Best Exact Accuracy %.3f " % (best_loss,))
		self.logger.info("Best Param:")
		self.task._print_param_dict(best_params)
		self.logger.info("-" * 50)


def parse_args(parser):
	parser.add_option("-m", "--model", type="string", dest="model_name", default='nfetc')
	parser.add_option("-d", "--data", type="string", dest="data_name", default='wikim')
	parser.add_option("-e", "--eval", type="int", dest="max_evals", default=100)
	parser.add_option("-c", "--cv_runs", type="int", dest="cv_runs", default=3)
	parser.add_option("-p", "--portion", type="int", dest="portion", default=100)
	parser.add_option("-s", "--savename", type="string", dest="save_name", default='')
	parser.add_option("-f", "--cvfull", dest="cvfull", default=False, action='store_true')

	options, args = parser.parse_args()
	return options, args


# python task.py -m <model_name> -d <data_name> -e <max_evals> -c <cv_runs>
def main(options):
	time_str = datetime.datetime.now().isoformat()
	logname = "[Model@%s]_[Data@%s]_[use@%s]_%s.log" % (options.model_name, options.data_name,options.save_name, time_str)
	logger = logging_utils._get_logger(config.LOG_DIR, logname)
	optimizer = TaskOptimizer(options.model_name, options.data_name, options.cv_runs,
	                          options.max_evals, logger, options.cvfull, options.save_name,options.portion)
	optimizer.run()


if __name__ == "__main__":
	parser = OptionParser()
	options, args = parse_args(parser)
	main(options)
