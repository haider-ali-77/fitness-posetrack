import numpy as np
import cv2


def get_exp_weighed_avg(buff,start_index):
	pts = np.zeros(buff.shape[1:])
	counter = 0
	decay  = 3
	sum = np.sum(np.exp(-decay*np.arange(0,len(buff))))
	for i in range(start_index,len(buff)+start_index):
		curr_ind = i % len(buff)

		pts+= np.exp(-decay*counter)*buff[curr_ind]
		counter+=1
	return pts/sum




vpts = np.load('points.npy')


cap = cv2.VideoCapture('trisha_right_20ft_slomo_IMG_2495.mov')


h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('averaging.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))

k_filter = cv2.KalmanFilter()
buf_size = 30
len_pts = vpts.shape[1]
pts_buf = np.zeros((buf_size,len_pts))

counter = 0


for pts in vpts:
	if counter == 0:
		for buf_i in range(0,buf_size):
			pass
			pts_buf[buf_i] = pts

		# old_pts=pts
	old_pts = pts.copy().reshape(-1,2)
	curr_ind = counter % buf_size
	pts_buf[curr_ind] = pts
	pts = get_exp_weighed_avg(pts_buf,curr_ind)
	pts = pts.reshape(-1,2)

# pts = old_pts*0.6 +pts*0.4
# 	old_pts = pts
	pts = pts.astype(np.int)


	ret,frame = cap.read()
	frame2 = frame.copy()
	old_pt = pts[0]
	for pt in pts:
		cv2.circle(frame,tuple(pt),3,(0,255,0),cv2.FILLED)
		cv2.line(frame,tuple(old_pt),tuple(pt),(255,255,255))
		old_pt= pt
	for pt in old_pts:
		cv2.circle(frame2,tuple(pt),5,(255,0,0),cv2.FILLED)
	counter+=1
	writer.write(frame)
	# frame = cv2.vconcat((frame,frame2))
	# frame = cv2.resize(frame,(1280,720))
	# cv2.imshow("vid",frame)
	# cv2.waitKey(30)

	if not ret:
		break
writer.release()
