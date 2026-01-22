import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy


def smooth_hip_points(video_keypoints,source_fps=60,slow_mo =False):
	"""

	:param video_keypoints: expected shape (N,17,2) where N is the number of frames
	:return: modified video_keypoints
	"""
	video_keypoints = video_keypoints[:, :, :2]
	# video_keypoints = savgol_filter(video_keypoints.reshape(-1, 34), polyorder=3, window_length=21, axis=0).reshape(-1, 17, 2)
	hip_ptsv = video_keypoints[:, 11:13, :]
	win_len = int(8*source_fps/60)
	if slow_mo:
		win_len = win_len*2 -1
	win_len+= (not win_len%2)
	print('smoothing window length: ',win_len)
	poly_order = 3
	if win_len <=3:
		poly_order = win_len -1
	elif win_len >9:
		poly_order = win_len -5
	loop_len=win_len
	while loop_len >0:
		hip_ptsv = savgol_filter(hip_ptsv.reshape(-1, 4), polyorder=2, window_length=5, axis=0).reshape(-1, 2,2)
		loop_len-=5

	def get_hip_avg_kernel_size(sig):
		pk1 = scipy.signal.find_peaks(sig)[0]
		# plt.scatter(x=pk1,y=hip_ptsv[pk1,0,1])
		delta_pk1 = np.abs(np.diff(sig[pk1], append=0))
		delta_pk1[-1] = 0
		n1 = np.argmax(delta_pk1)
		return int((pk1[n1 + 1] - pk1[n1]) / 3)

	hip_ptsv_savgol = hip_ptsv.copy()
	# print(pk1)
	kernel_size = get_hip_avg_kernel_size(hip_ptsv[:, 0, 1]) + get_hip_avg_kernel_size(hip_ptsv[:, 1, 1])
	kernel_size = int(kernel_size / 2)
	if kernel_size > len(video_keypoints):
		kernel_size = int(len(video_keypoints)/10)+1
	if kernel_size == 0:
		kernel_size=int(win_len/4)+1
	print('averaging kernel size: ',kernel_size)
	avg_kernel = np.ones(kernel_size) / kernel_size
	hip_ptsv[:, 0, 1] = np.convolve(hip_ptsv[:, 0, 1], avg_kernel, 'same')
	hip_ptsv[:, 1, 1] = np.convolve(hip_ptsv[:, 1, 1], avg_kernel, 'same')
	hip_ptsv[:kernel_size] = hip_ptsv_savgol[:kernel_size]
	hip_ptsv[-kernel_size:] = hip_ptsv_savgol[-kernel_size:]

	video_keypoints[:, 11:13, :] = hip_ptsv
	# video_keypoints[:, 11:13, :] = hip_ptsv_savgol

	return video_keypoints


if __name__=='__main__':
	#hip demo
	cap = cv2.VideoCapture('trisha_right_20ft_slomo_IMG_2495.mov')
	ptsv = np.load('trisha_right.npy')
	ptsv = smooth_hip_points(ptsv)
	for frame_ctr,pts in enumerate(ptsv):
		pts = pts.astype(np.int)
		ret,frame = cap.read()

		for i,pt in enumerate(pts):

			if i == 11 or i==12:
				cv2.circle(frame,tuple(pt),4,(0,0,255),cv2.FILLED)

				cv2.circle(frame,tuple(pt),5,(0,255,255),cv2.FILLED)


			cv2.circle(frame,tuple(pt),3,(0,255,0),cv2.FILLED)

		frame = cv2.resize(frame,dsize=None,fx=0.7,fy=0.7)

		cv2.imshow("vid",frame)
		cv2.waitKey(20)
	cv2.destroyWindow("vid")