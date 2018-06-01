import tensorflow as tf
import numpy as np
from datetime import datetime
from absl import flags
from absl import app
import matplotlib.pyplot as plt

import utils


FLAGS = flags.FLAGS

flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_string("save_path", "models/", "Save path for models")

class DAGMM(object):

	def __init__(self, sess, input_dim=6, gmm_k=2, lmda_1=0.1, lmda_2=0.005, name="dagmm"):
		super(DAGMM, self).__init__()
		self.name = name
		self.input_dim = input_dim
		self.gmm_k = gmm_k
		self.lmda_1 = lmda_1
		self.lmda_2 = lmda_2
		with tf.variable_scope(self.name):
			self.build_model()
			self.sess = sess
			self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name), max_to_keep=3)

	def build_model(self):

		# Compression Network
		# Takes x
		# Produces x'
		# Produces z = concat((z_c, z_r)) (Equations 1, 2, 3)
		# z_r = concat((eu_dist, cos_sim))
		self.input = tf.placeholder(
			shape=(None, self.input_dim),
			dtype=tf.float32,
			name="input",
		)
		encoder_1 = tf.layers.dense(
			inputs=self.input,
			units=12,
			activation=tf.tanh,
		)
		encoder_2 = tf.layers.dense(
			inputs=encoder_1,
			units=4,
			activation=tf.tanh,
		)
		self.z_c = tf.layers.dense(
			inputs=encoder_2,
			units=1,
			activation=None,
		)
		decoder_1 = tf.layers.dense(
			inputs=self.z_c,
			units=4,
			activation=tf.tanh,
		)
		decoder_2 = tf.layers.dense(
			inputs=decoder_1,
			units=12,
			activation=tf.tanh,
		)
		self.recon = tf.layers.dense(
			inputs=decoder_2,
			units=self.input_dim,
			activation=None,
		)
		
		eu_dist = tf.norm(self.input - self.recon, axis=1, keep_dims=True) / tf.norm(self.input, axis=1, keep_dims=True)
		cos_sim = tf.reduce_sum(self.input * self.recon, axis=1, keep_dims=True) / (tf.norm(self.input, axis=1, keep_dims=True) * tf.norm(self.recon, axis=1, keep_dims=True))
		self.z_r = tf.concat((eu_dist, cos_sim), axis=1)

		self.z = tf.concat((self.z_c, self.z_r), axis=1)
		
		# Estimation Network
		# Takes z = concat((z_c, z_r))
		# Produces p, where gamma = softmax(p) = soft mixture-component membership prediction (Equation 4)
		self.is_train = tf.placeholder(
			# for dropout
			shape=None,
			dtype=tf.bool,
			name="is_train",
		)
		estim_1 = tf.layers.dense(
			inputs=self.z,
			units=10,
			activation=tf.tanh,
		)
		estim_dropout = tf.layers.dropout(
			inputs=estim_1,
			rate=0.5,
			training=self.is_train,
		)
		self.p = tf.layers.dense(
			inputs=estim_dropout,
			units=self.gmm_k,
			activation=None,
		)
		self.gamma = tf.nn.softmax(self.p)

		# GMM parameters: gmm_dist (phi), gmm_mean (mu), gmm_cov (epsilon) (Equation 5)
		# self.gmm_dist = tf.expand_dims(tf.reduce_mean(self.gamma, axis=0, keep_dims=True), axis=2)
		self.gmm_dist = tf.transpose(tf.reduce_mean(self.gamma, axis=0, keep_dims=True))
		self.gmm_mean = tf.matmul(self.gamma, self.z, transpose_a=True) / tf.transpose(tf.reduce_sum(self.gamma, axis=0, keep_dims=True))
		self.diff_mean = diff_mean = tf.tile(tf.expand_dims(self.z, axis=0), tf.constant([self.gmm_k, 1, 1])) - tf.expand_dims(self.gmm_mean, axis=1)
		self.gmm_cov = tf.matmul(tf.transpose(diff_mean, perm=[0, 2, 1]), tf.expand_dims(tf.transpose(self.gamma), axis=2) * diff_mean) / tf.expand_dims(tf.transpose(tf.reduce_sum(self.gamma, axis=0, keep_dims=True)), axis=2)
		# Energy Function (Equation 6)
		energy_numerator = tf.exp(-0.5 * tf.reduce_sum(tf.matmul(self.diff_mean, self.gmm_cov) * self.diff_mean, axis=2))
		energy_denominator = tf.expand_dims(tf.expand_dims(tf.sqrt(tf.matrix_determinant(2 * np.pi * self.gmm_cov)), axis=1), axis=2)
		self.energy = tf.expand_dims(-tf.log(tf.reduce_sum(tf.reduce_sum(tf.expand_dims(self.gmm_dist, axis=1) * energy_numerator / energy_denominator, axis=0), axis=0)), axis=1)

		# Loss Function (Equation 7)
		# Reconstruction loss + lmda_1 * Energy loss + lmda_2 * Diagonal loss
		# self.recon_loss = recon_loss = tf.losses.mean_squared_error(self.input, self.recon)
		self.recon_loss = recon_loss = tf.reduce_mean(tf.norm((self.input - self.recon), axis=1) ** 2)
		self.energy_loss = energy_loss = tf.reduce_mean(self.energy)
		self.diagonal_loss = diagonal_loss = tf.reduce_sum(tf.pow(tf.matrix_diag_part(self.gmm_cov), -tf.ones_like(tf.matrix_diag_part(self.gmm_cov))))
		self.loss = recon_loss + self.lmda_1 * energy_loss + self.lmda_2 * diagonal_loss

		self.optimize = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
		
	def save(self, save_path, global_step=None):
		self.saver.save(self.sess, save_path=save_path + "checkpoint", global_step=global_step)
		print("Model saved to {}.".format(save_path))

	def load(self, save_path):
		ckpt = tf.train.latest_checkpoint(save_path)
		self.saver.restore(self.sess, ckpt)
		print("Model loaded from {}.".format(save_path))

	def fit(self, x, epochs=20000, batchsize=1024):
		iterations = int(len(x) / batchsize)
		for i in np.arange(epochs):
			start_time = datetime.now()
			for j in np.arange(iterations):
				start = int(j * batchsize)
				end = int(start + batchsize)
				if end > len(x):
					x_minibatch = np.concatenate((x[start:], x[:int(end - len(x))]))
				else:
					x_minibatch = x[start:end]
				feed_dict = {
					self.input: x_minibatch,
					self.is_train: True,
				}
				loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=feed_dict)
			if (i + 1) % 500 == 0:
				duration = datetime.now() - start_time
				print("Epoch {}/{} - Loss: {:.3f} - Time per Epoch: {}".format(i + 1, epochs, loss, duration))
				self.save(FLAGS.save_path, i)
		print("Epoch {}/{} - Loss: {:.3f} - Time per Epoch: {}".format(i + 1, epochs, loss, duration))
		print("Training complete!")
		self.save(FLAGS.save_path, i)

	def predict(self, x, contamination=0.025):
		# Calculate energy for each sample
		# Assign top {{ contamination * 100% }} samples as anomalies
		# Set anomalies as -1
		# Set nominals as +1
		feed_dict = {
			self.input: x,
			self.is_train: False,
		}
		energy = self.sess.run(self.energy, feed_dict=feed_dict)
		sorted_energy = np.copy(energy).reshape(-1)
		sorted_energy.sort()
		
		fig,ax = plt.subplots()
		ax.plot(sorted_energy, np.arange(len(sorted_energy)))
		plt.show()
		
		threshold = sorted_energy[int(len(x) * (1 - contamination))]
		predictions = -2 * (energy.reshape(-1) >= threshold).astype(np.int32) + 1
		return predictions

def main(unused_args):
	dataset = utils.ThyroidData()
	(x_train, y_train), (x_test, y_test) = dataset.load_data()
	if FLAGS.train:
		tf.gfile.MakeDirs(FLAGS.save_path)
		with tf.Session() as sess:
			dagmm = DAGMM(sess)
			sess.run(tf.global_variables_initializer())
			dagmm.fit(x_train)
	elif FLAGS.test:
		with tf.Session() as sess:
			dagmm = DAGMM(sess)
			dagmm.load(FLAGS.save_path)
			predictions = dagmm.predict(x_test)
			precision, recall, f1 = utils.get_stats(predictions=predictions, labels=y_test)
			print("Precision: {}".format(precision))
			print("Recall: {}".format(recall))
			print("F1: {}".format(f1))


if __name__ == "__main__":
	app.run(main)