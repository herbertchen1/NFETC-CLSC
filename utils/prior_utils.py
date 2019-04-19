from utils import pkl_utils
import sys
sys.path.append("../")
import config
import pprint
import numpy as np
import collections,copy

#每一个列代表了一个type,其子类型有一个alpha控制loss权重,return ntype,ntype矩阵
def create_prior(type_info, alpha=1.0):
	#print(type_info)
	type2id, typeDict = pkl_utils._load(type_info)
	#pprint.pprint(typeDict)
	num_types = len(type2id)
	prior = np.zeros((num_types, num_types))
	for x in type2id.keys():
		tmp = np.zeros(num_types)
		tmp[type2id[x]] = 1.0
		for y in typeDict[x]:#子节点
			tmp[type2id[y]] = alpha
		#print(tmp)
		prior[:,type2id[x]] = tmp
	return prior
def istopSon(s)->bool:
	istop=False
	counter=collections.Counter(s)
	if counter['/']==1:
		istop=True
	return istop

def makeSonFindermatrix(type_info):
	# print(type_info)
	type2id, typeDict = pkl_utils._load(type_info)
	#pprint.pprint(typeDict)
	num_types = len(type2id)
	prior = np.zeros((num_types, num_types))
	for x in type2id.keys():
		tmp = np.zeros(num_types)
		tmp[type2id[x]] = 1.0
		for y in typeDict[x]:  # 子节点
			tmp[type2id[y]] = 1.0
		prior[type2id[x],:] = tmp
	#print('-'*50)
	tmp = np.zeros(num_types)
	for typename in typeDict.keys():
		if istopSon(typename):
			# print(typename)
			# print(type2id[typename])
			tmp[type2id[typename]]=1
	# prior[num_types,num_types]=1
	fatherNotin=copy.deepcopy(prior)
	for i in range(num_types):
		fatherNotin[i,i]=0
	#print('-' * 50)
	# id2type={val:key for key,val in type2id.items()}
	# for j,i in enumerate(prior[num_types,:]):
	# 	if i==1:
	# 		print(id2type[j])
	return prior,fatherNotin,tmp

if __name__ == '__main__':
    type_info='.'+config.WIKIM_TYPE
    print(type_info)

    #pprint.pprint(create_prior(type_info)[5,:])
    pprint.pprint(makeSonFindermatrix(type_info)[0][101,:])

