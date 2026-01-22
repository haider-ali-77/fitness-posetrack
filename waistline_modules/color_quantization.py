import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

def cluster_quantization(image,num_clusters = 2,clt=None):
	(h, w) = image.shape[:2]
	# convert the image from the RGB color space to the L*a*b*
	# color space -- since we will be clustering using k-means
	# which is based on the euclidean distance, we'll use the
	# L*a*b* color space where the euclidean distance implies
	# perceptual meaning
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	# reshape the image into a feature vector so that k-means
	# can be applied
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	# apply k-means using the specified number of clusters and
	# then create the quantized image based on the predictions
	if clt is None:
		clt = MiniBatchKMeans(n_clusters = num_clusters)
	clt = clt.partial_fit(image)
	labels =clt.predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]
	# reshape the feature vectors to images
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))
	# convert from L*a*b* to RGB
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
	# display the images and wait for a keypress
	return quant,clt