import scipy.io
import numpy as np


def get_stats(predictions, labels):
	# anomalies are labeled -1; we treat anomalies as positives
	tp = np.sum((predictions == labels)[np.where(predictions == -1)])
	tn = np.sum((predictions == labels)[np.where(predictions == 1)])
	fp = np.sum((predictions != labels)[np.where(predictions == -1)])
	fn = np.sum((predictions != labels)[np.where(predictions == 1)])

	if tp + fp == 0.:
		precision = 1.
	else:
		precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	if precision + recall == 0.:
		f1 = 0.
	else:
		f1 = 2 * precision * recall / (precision + recall)

	return precision, recall, f1

class ThyroidData(object):

	def __init__(self):
		super(ThyroidData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("data/thyroid.mat")
		samples = data['X'] # 3772 samples
		labels = (-2 * (data['y'] == 1).astype(np.int32) + 1).reshape(-1)

		# normalize
		# data_max = np.max(samples, axis=0)
		# data_min = np.min(samples, axis=0)
		# samples -= data_min
		# samples /= data_max - data_min
		# samples -= 0.5

		nominal_samples = samples[np.where(labels == 1)] # 3679 nominal
		nominal_labels = labels[np.where(labels == 1)]
		anomalous_samples = samples[np.where(labels == -1)] # 93 anomalies
		anomalous_labels = labels[np.where(labels == -1)]

		n_train = int(len(nominal_samples) / 2)
		n_anomalous_test = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train] # 1839 training
		y_train = nominal_labels[:n_train]

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_anomalous_test:])) # 1933 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_anomalous_test:]))

		return (x_train, y_train), (x_test, y_test)

class ArrhythmiaData(object):

	def __init__(self):
		super(ArrhythmiaData, self).__init__()

	def load_data(self):
		data = scipy.io.loadmat("data/arrhythmia.mat")
		samples = data['X'] # 518 samples
		labels = (-2 * (data['y'] == 1).astype(np.int32) + 1).reshape(-1)

		nominal_samples = samples[np.where(labels == 1)] # 452 nominal
		nominal_labels = labels[np.where(labels == 1)]
		anomalous_samples = samples[np.where(labels == -1)] # 66 anomalies
		anomalous_labels = labels[np.where(labels == -1)]

		n_train = int(len(nominal_samples) / 2)
		n_anomalous_test = int(len(anomalous_samples) / 2)

		x_train = nominal_samples[:n_train] # 226 training
		y_train = nominal_labels[:n_train]

		x_test = np.concatenate((nominal_samples[n_train:], anomalous_samples[n_anomalous_test:])) # 259 test
		y_test = np.concatenate((nominal_labels[n_train:], anomalous_labels[n_anomalous_test:]))

		return (x_train, y_train), (x_test, y_test)
