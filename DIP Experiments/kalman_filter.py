from pykalman import KalmanFilter
import numpy as np
import cv2


vpts = np.load('points.npy')
# vpts = vpts.reshape(len(vpts),-1,2)

cap = cv2.VideoCapture('trisha_right_20ft_slomo_IMG_2495.mov')
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))
writer2 = cv2.VideoWriter('kalman.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))

buff_len = 40
kf = KalmanFilter(n_dim_obs=34,n_dim_state=34,initial_state_mean=vpts[0])
kf = kf.em(vpts[0:buff_len],n_iter=5)
for i in range(0,buff_len):
	cap.read()
for i in range(buff_len,len(vpts)):
	ret,frame = cap.read()
	mean,cov = kf.filter(vpts[0:i])
	next_mean,next_cov = kf.filter_update(mean[-1],cov[-1],observation=vpts[i])
	pts = next_mean.astype(np.int).reshape(-1,2)
	# print('vpts:',vpts[i].reshape(-1,2))
	# print('pts',pts)
	# print('-'*30)
	# print(vpts[i],next_mean)
	frame2=frame.copy()
	for pt in vpts[i].reshape(-1,2):
		cv2.circle(frame,tuple(pt),3,(0,255,0),cv2.FILLED)
	for pt in pts:
		cv2.circle(frame2,tuple(pt),3,(0,255,0),cv2.FILLED)
	# frame = cv2.resize(frame,(1280,720))
	# cv2.imshow("vid",frame)
	# cv2.waitKey(30)
	writer.write(frame)
	writer2.write(frame2)


writer.release()
writer2.release()