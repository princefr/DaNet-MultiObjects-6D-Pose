from tensorflow.python.keras import backend as K
from tensorflow.python import keras

import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, :, 4 * num_anchors:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

		return lambda_rpn_regr * K.sum(
			y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])


	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * keras.losses.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :])



def pose_loss(target_class_ids):
	def regression_pose_loss(y_true, y_pred):
		"""
		:param y_true:
		:param y_pred:
		:return:
		"""
		# Reshape to merge batch and roi dimensions for simplicity.
		target_class_ids = K.reshape(target_class_ids, (-1,))
		target_pose = K.reshape(y_true, (-1, 4))
		pred_pose = K.reshape(y_pred, (-1, K.int_shape(y_pred)[2], 4))

		# Only positive ROIs contribute to the loss. And only
		# the right class_id of each ROI. Get their indicies.
		positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
		positive_roi_class_ids = tf.cast(
			tf.gather(target_class_ids, positive_roi_ix), tf.int64)
		indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

		# Gather the deltas (predicted and true) that contribute to loss
		target_pose = tf.gather(target_pose, positive_roi_ix)
		pred_pose = tf.gather_nd(pred_pose, indices)

		# Smooth-L1 Loss
		loss = K.switch(tf.size(target_pose) > 0,
						K.square(target_pose - pred_pose),
						tf.constant(0.0))
		loss = K.mean(loss)
		return loss
	return regression_pose_loss
