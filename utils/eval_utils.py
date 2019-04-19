import copy,json,os
def f1(p, r):
	if p == 0. or r == 0.:
		return 0.
	return 2*p*r/(p+r)

def label_path(t):
	types = t.split("/")
	if len(types) == 3:
		return ["/"+types[1], t]
	if len(types) == 4:
		return ["/"+types[1], "/"+types[1]+"/"+types[2], t]
	return [t]

def complete_path(t):
	v = []
	for x in t:
		v.extend(label_path(x))
	return set(v)

def strict(labels, predictions,oridata=None,modelname=''):
	cnt = 0
	if not modelname:
		outfile='./answer/duibi_test.json'
	else:
		answerdir=os.path.join('answer', modelname)
		if not os.path.isdir(answerdir):
			os.makedirs(answerdir)
		outfile = os.path.join(answerdir,'result.json')
	print("len:",len(labels),len(predictions))
	for i,(label, pred) in enumerate(zip(labels, predictions)):
		cnt += set(label) == set(pred)

	acc = cnt/len(labels)
	print("Strict Accuracy: %s" % acc)

	return acc

def loose_macro(labels, predictions):
	p = 0.
	r = 0.
	for label, pred in zip(labels, predictions):
		label = set(label)
		pred = set(pred)
		if len(pred) > 0:
			p += len(label.intersection(pred))/len(pred)
		if len(label) > 0:
			r += len(label.intersection(pred))/len(label)
	p /= len(labels)#所有样本平均的p
	r /= len(labels)#平均的r
	f = f1(p, r)
	print("Loose Macro:")
	print("Precision %s Recall %s F1 %s" % (p, r, f))
	return p, r, f

def loose_micro(labels, predictions):
	cnt_pred = 0
	cnt_label = 0
	cnt_correct = 0
	for label, pred in zip(labels, predictions):
		label = set(label)
		pred = set(pred)
		cnt_pred += len(pred)
		cnt_label += len(label)
		cnt_correct += len(label.intersection(pred))
	p = cnt_correct/cnt_pred
	r = cnt_correct/cnt_label
	f = f1(p, r)#所有东西加起来算f1
	print("Loose Micro:")
	print("Precision %s Recall %s F1 %s" % (p, r, f))
	return p, r, f
