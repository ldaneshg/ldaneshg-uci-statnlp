#!/bin/python
import scipy.sparse as sp
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#import tensorflow_text as tf
from scipy.fftpack import fft

def read_files(tarfname):
	"""Read the training and development data from the speech tar file.
	The returned object contains various fields that store the data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	speech = Data()
	print("-- train data")
	speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
	print(len(speech.train_data))
	print("-- dev data")
	speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
	print(len(speech.dev_data))
	print("-- transforming data and labels")
	regex1 = '[a-zA-Z]{1,8}'
	speech.count_vect = CountVectorizer(token_pattern=regex1)#stop_words="english")#, binary=True)
	#speech.count_vect.ngram_range = (1, 6)
	speech.trainX = speech.count_vect.fit_transform(speech.train_data)
	#tfidf = TfidfTransformer()
	#temp = tfidf.fit_transform(speech.trainX)
	#speech.trainX = sp.hstack((speech.trainX.astype('float64'), temp), format='csr')
	speech.devX = speech.count_vect.transform(speech.dev_data)
	#temp2 = tfidf.fit_transform(speech.devX)
	#speech.devX = sp.hstack((speech.devX.astype('float64'), temp2), format='csr')
	from sklearn import preprocessing
	speech.le = preprocessing.LabelEncoder()
	speech.le.fit(speech.train_labels)
	speech.target_labels = speech.le.classes_
	speech.trainy = speech.le.transform(speech.train_labels)
	speech.devy = speech.le.transform(speech.dev_labels)

	testing = np.zeros((19,speech.trainX.shape[1]))
	test2 = np.zeros((19,speech.devX.shape[1]))
	for j in range(19):
		testing[j,:] = np.sum(speech.trainX[np.where(speech.trainy == j)[0]], axis=0)
		#test2[j,:] = np.sum(speech.devX[np.where(speech.devy == j)[0]], axis=0)
	que = np.matmul(speech.trainX.toarray(),testing.T)/10
	#speech.trainX = sp.hstack((speech.trainX, que), format='csr')
	que2 = np.matmul(speech.devX.toarray(), testing.T)/10
	#speech.devX = sp.hstack((speech.devX, que2), format='csr')
	#speech.trainX = que/10
	#speech.devX = que2/10

	tar.close()
	return speech

def read_unlabeled(tarfname, speech):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the speech.vectorizer
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = speech.count_vect.transform(unlabeled.data)
	print(unlabeled.X.shape)
	tar.close()
	return unlabeled

def read_tsv(tar, fname):
	member = tar.getmember(fname)
	print(member.name)
	tf = tar.extractfile(member)
	data = []
	labels = []
	fnames = []
	for line in tf:
		line = line.decode("utf-8")
		(ifname,label) = line.strip().split("\t")
		#print ifname, ":", label
		content = read_instance(tar, ifname)
		labels.append(label)
		fnames.append(ifname)
		data.append(content)
	return data, fnames, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the speech object,
	this function write the predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The speech object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = speech.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	for i in range(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		# iid = file_to_id(fname)
		f.write(str(i+1))
		f.write(",")
		#f.write(fname)
		#f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()

def file_to_id(fname):
	return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			# iid = file_to_id(ifname)
			i += 1
			f.write(str(i))
			f.write(",")
			#f.write(ifname)
			#f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts OBAMA_PRIMARY2008 for all the instances.
	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("OBAMA_PRIMARY2008")
			f.write("\n")
	f.close()

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

if __name__ == "__main__":
	print("Reading data")
	tarfname = "data/speech.tar.gz"
	speech = read_files(tarfname)
	print("Training classifier")
	import classify

	"""for i in range(10):
		exclude = speech.trainX[437*(i):437*(i+1),:]
		excl_labels = speech.trainy[437*(i):437*(i+1)]
		include = speech.trainX.copy()
		incl_labels = speech.trainy.copy()
		mask = np.ones(include.shape, dtype=bool)
		mask[437*(i):437*(i+1),:] = False
		include = include[np.where(mask.any(axis=1))]
		mask = np.ones(incl_labels.shape, dtype=bool)
		mask[437 * (i):437 * (i + 1)] = False
		incl_labels = incl_labels[mask]
		cls = classify.train_classifier(include, incl_labels)
		print(i)
		classify.evaluate(include, incl_labels, cls)
		#classify.evaluate(sp.vstack((speech.devX, exclude), format='csr'), np.concatenate([speech.devy, excl_labels]), cls)
		classify.evaluate(exclude, excl_labels, cls)"""

	print(speech.trainX.shape)
	#cls = classify.train_classifier(speech.trainX, speech.trainy)
	#speech.trainX = speech.trainX[:,np.where((np.absolute(cls.coef_) > 0.25 ).any(axis=0))[0]]
	#speech.devX = speech.devX[:,np.where((np.absolute(cls.coef_) > 0.25 ).any(axis=0))[0]]
	#print(speech.trainX.shape)
	cls = classify.train_classifier(speech.trainX, speech.trainy)
	#cls2 = sklearn.base.clone(cls)
	#cls2.fit(speech.trainX, speech.trainy)
	print("Evaluating")
	classify.evaluate(speech.trainX, speech.trainy, cls)
	classify.evaluate(speech.devX, speech.devy, cls)


	print("Reading unlabeled data")
	unlabeled = read_unlabeled(tarfname, speech)
	#tfidf = TfidfTransformer()
	#unlabeled.X = sp.hstack((unlabeled.X.astype('float64'), tfidf.fit_transform(unlabeled.X)), format='csr')
	#unlabeled.X = unlabeled.X[:, np.where((np.absolute(cls.coef_) > 0.40).any(axis=0))[0]]

	for i in range(5):

		samples = unlabeled.X[0:4000,:]
		prob = cls.predict_proba(samples)
		samples = samples[np.where((prob > (0.60)).any(axis=1))]
		pred_labels = cls.predict(samples)
		temp = np.concatenate([speech.trainy, pred_labels])
		cls = classify.train_classifier(sp.vstack((speech.trainX, samples), format='csr'), temp)
		#acc = classify.evaluate(speech.devX, speech.devy, cls)
		print(i)

	"""for i in range(3):
		samples = unlabeled.X[0:21671,:]
		samples2 = unlabeled.X[21671:43342,:]
		prob = cls.predict_proba(samples)
		prob2 = cls2.predict_proba(samples2)
		samples = samples[np.where((prob > (0.90)).any(axis=1))]
		#samples2 = samples2[np.where((prob2 > (0.90)).any(axis=1))]
		print(samples.shape)
		pred_labels = cls.predict(samples)
		pred_labels2 = cls2.predict(samples2)
		temp = np.concatenate([speech.trainy, pred_labels])
		temp2 = np.concatenate([speech.trainy, pred_labels2])
		cls = classify.train_classifier(sp.vstack((speech.trainX, samples), format='csr'), temp)
		cls2 = classify.train_classifier(sp.vstack((speech.trainX, samples2), format='csr'), temp2)
		prob = cls.predict_proba(unlabeled.X)
		prob2 = cls2.predict_proba(unlabeled.X)
		samples = unlabeled.X[np.where(((prob > (0.50)).any(axis=1)) & ((prob2 > (0.50)).any(axis=1)))]
		pred_labels = cls.predict(samples)
		temp = np.concatenate([speech.trainy, pred_labels])
		cls = classify.train_classifier(sp.vstack((speech.trainX, samples), format='csr'), temp)"""

	speech.trainX = sp.vstack((speech.trainX, samples), format='csr')
	speech.trainy = np.concatenate([speech.trainy, pred_labels])
	#cls = classify.train_classifier(speech.trainX, temp)
	print("Evaluating")
	classify.evaluate(speech.trainX, speech.trainy, cls)
	classify.evaluate(speech.devX, speech.devy, cls)
	print("Writing pred file")
	write_pred_kaggle_file(unlabeled, cls, "data/speech-pred.csv", speech)
	coef = cls.coef_[0]

	# You can't run this since you do not have the true labels
	# print "Writing gold file"
	# write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
	# write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")
