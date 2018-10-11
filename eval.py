import numpy as np


def per_class_iou_acc(preds, labels, classes=8):
	assert isinstance(preds, np.ndarray) and isinstance(labels, np.ndarray)
	assert (len(preds.shape) == 1) and preds.shape == labels.shape

	classes_ious = {}
	classes_accs = {}
	for c_idx in range(classes):
		# Insert 
		iou_i = np.where(
				(preds == c_idx) & (labels == c_idx)
			)[0].shape[0]

		# Union
		iou_u = np.where(
				(preds == c_idx) | (labels == c_idx)
			)[0].shape[0]

		# Gt labels
		gts = np.where(
				labels == c_idx
			)[0].shape[0]

		classes_ious[c_idx] = (iou_i, iou_u)
		classes_accs[c_idx] = (iou_i, gts)

	return classes_ious, classes_accs


if __name__ == '__main__':
	preds = np.zeros((10, ))
	preds[0] = 1
	preds[5] = 4

	labels = np.ones((10, ))

	ious, accs = per_class_iou_acc(preds, labels, classes=8)

	print('ious: {}'.format(ious))
	print('accs: {}'.format(accs))
