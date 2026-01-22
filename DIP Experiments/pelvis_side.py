import numpy as np
import cv2
import scipy
from scipy.signal import savgol_filter
from waistline import findwaistline
import matplotlib.pyplot as plt
import keypoints_post_processing
import math
from scipy import interpolate


def pt_in_bbox(pt, bbox):
	bx1, by1, bx2, by2 = bbox
	if bx1 <= pt[0] <= bx2 and by1 <= pt[1] <= by2:
		return True
	return False


def intersectLines(pt1, pt2, ptA, ptB):
	""" this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

		returns a tuple: (xi, yi, valid, r, s), where
		(xi, yi) is the intersection
		r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
		s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
			valid == 0 if there are 0 or inf. intersections (invalid)
			valid == 1 if it has a unique intersection ON the segment    """

	DET_TOLERANCE = 0.00000001

	# the first line is pt1 + r*(pt2-pt1)
	# in component form:
	x1, y1 = pt1;
	x2, y2 = pt2
	dx1 = x2 - x1;
	dy1 = y2 - y1

	# the second line is ptA + s*(ptB-ptA)
	x, y = ptA;
	xB, yB = ptB;
	dx = xB - x;
	dy = yB - y;

	# we need to find the (typically unique) values of r and s
	# that will satisfy
	#
	# (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
	#
	# which is the same as
	#
	#    [ dx1  -dx ][ r ] = [ x-x1 ]
	#    [ dy1  -dy ][ s ] = [ y-y1 ]
	#
	# whose solution is
	#
	#    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
	#    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
	#
	# where DET = (-dx1 * dy + dy1 * dx)
	#
	# if DET is too small, they're parallel
	#
	DET = (-dx1 * dy + dy1 * dx)

	if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

	# now, the determinant should be OK
	DETinv = 1.0 / DET

	# find the scalar amount along the "self" segment
	r = DETinv * (-dy * (x - x1) + dx * (y - y1))

	# find the scalar amount along the input line
	s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

	# return the average of the two descriptions
	xi = (x1 + r * dx1 + x + s * dx) / 2.0
	yi = (y1 + r * dy1 + y + s * dy) / 2.0
	return (xi, yi, 1, r, s)


cap = cv2.VideoCapture('trisha_right_20ft_slomo_IMG_2495.mov')
# writer = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
writer2 = cv2.VideoWriter('pelvis_side.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (w, h))
keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle"]

keypoints_map = {i: j for j, i in enumerate(keypoints)}
print(keypoints_map)
ptsv = np.load('trisha_right.npy')
ptsv = keypoints_post_processing.smooth_hip_points(ptsv).astype(np.int)
# ptsv = savgol_filter(ptsv,polyorder=8,window_length=21,axis=0)

buf_len = 21

clt = None
counter = 0
ly1 = []
ly2 = []
angles = []
for pts in ptsv:
	pts = pts.reshape(-1, 2)
	ret, frame = cap.read()
	mid_chest = pts[keypoints_map['left_shoulder']] + pts[keypoints_map['right_shoulder']]
	mid_chest = mid_chest / 2
	mid_hip = pts[keypoints_map['left_hip']] + pts[keypoints_map['left_hip']]
	mid_hip = mid_hip / 2
	back_length = np.abs(mid_hip[1] - mid_chest[1])
	bbox_w = back_length / 6
	mode = 'right'
	x1 = pts[keypoints_map[mode + '_hip']][0] - bbox_w / 2
	x2 = pts[keypoints_map[mode + '_hip']][0] + bbox_w / 2
	y1 = mid_chest[1] + back_length * 0.5
	y2 = pts[keypoints_map[mode + '_hip']][1]
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

	crop = frame[y1:y2, x1:x2]

	img = crop

	# img  = cv2.GaussianBlur(img,(5,5),1)
	angle, aline, sline, clt = findwaistline(img, 3, clt)

	rsz_fac = 128 / min(img.shape[0], img.shape[1])
	rsz_fac = 1
	img = cv2.resize(img, dsize=None, fx=rsz_fac, fy=rsz_fac)
	aline = [int(pt * rsz_fac) for pt in aline]
	sline = [int(pt * rsz_fac) for pt in sline]

	pt1 = tuple(aline[0:2])
	pt2 = tuple(aline[2:])
	cv2.arrowedLine(img, pt1, pt2, (0, 255, 255), 2, tipLength=0.02)
	pt1 = tuple(sline[0:2])
	pt2 = tuple(sline[2:])
	cv2.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.02)
	cv2.circle(img, pt1, 5, (0, 255, 0), cv2.FILLED)

	ly1.append(aline[1])
	ly2.append(aline[3])
	angles.append(angle)
	cv2.imwrite('outputs/' + str(counter) + '.png', img)

	# img = cv2.resize(img,(512,512))
	# cv2.imshow('Angle in Degrees: ' "{:.2f}".format(angle), img)
	# cv2.imshow('vid', img)
	# counter+=1
	#
	# cv2.waitKey(100)
	#

	writer2.write(frame)
writer2.release()
# %%
plt.plot(angles)
plt.title('angle')
plt.show()

plt.plot(ly1)
plt.plot(ly2)
plt.legend(['y1', 'y2'])
plt.title('Y coords')
plt.show()
# %%
post_process_angle = False
if post_process_angle:
	# ly1 = savgol_filter(ly1,15,3)
	# ly2 = savgol_filter(ly1,15,3)
	dly1 = np.abs(np.diff(ly1, prepend=ly1[0]))
	dth_ly1 = np.average(dly1[dly1 > np.median(dly1)])
	dly2 = np.abs(np.diff(ly2, prepend=ly2[0]))
	dth_ly2 = np.average(dly2[dly2 > np.median(dly2)])

	print(dth_ly1, dth_ly2)
	for i in range(1, len(ly1)):
		if abs(ly1[i] - ly1[i - 1]) > dth_ly1:
			ly1[i] = ly1[i - 1]
		if abs(ly2[i] - ly2[i - 1]) > dth_ly2:
			ly2[i] = ly2[i - 1]

	# sos = scipy.signal.butter(1,0.2,'low',output='sos')
	# ly1 = scipy.signal.sosfilt(sos,ly1)
	# ly2 = scipy.signal.sosfilt(sos,ly2)

	angles = savgol_filter(angles, 21, 3)

	plt.plot(angles)
	plt.title('angle')
	plt.show()

	plt.plot(ly1)
	plt.plot(ly2)
	plt.legend(['y1', 'y2'])
	plt.title('Y coords')
	plt.show()

# %%

# run to check post processing results
writer2 = cv2.VideoWriter('pelvis_side_post.avi', cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (w, h))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for ctr, pts in enumerate(ptsv):
	pts = pts.reshape(-1, 2)
	ret, frame = cap.read()
	mid_chest = pts[keypoints_map['left_shoulder']] + pts[keypoints_map['right_shoulder']]
	mid_chest = mid_chest / 2
	mid_hip = pts[keypoints_map['left_hip']] + pts[keypoints_map['left_hip']]
	mid_hip = mid_hip / 2
	back_length = np.abs(mid_hip[1] - mid_chest[1])
	bbox_w = back_length / 6
	mode = 'right'
	x1 = pts[keypoints_map[mode + '_hip']][0] - bbox_w / 2
	x2 = pts[keypoints_map[mode + '_hip']][0] + bbox_w / 2
	y1 = mid_chest[1] + back_length * 0.5
	y2 = pts[keypoints_map[mode + '_hip']][1]
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	arm_kpts = [keypoints_map['right_elbow'],keypoints_map['right_wrist']]

	crop = frame[y1:y2, x1:x2].copy()

	arm_pts = pts[arm_kpts]
	f_arm = interpolate.interp1d([0,10],arm_pts,axis=0)
	arm_pts = f_arm(np.arange(0,10))
	valid_angle = True
	for pt in arm_pts:
			pt = pt.astype(np.int)
			if pt_in_bbox(pt,[x1,y1,x2,y2]):
				cv2.circle(frame,(0,0),100,(0,0,255),cv2.FILLED)
				cv2.circle(frame,tuple(pt),3,(0,0,255),cv2.FILLED)
				valid_angle = False
			else:
				cv2.circle(frame,tuple(pt),3,(0,255,0),cv2.FILLED)






	img = crop

	# img  = cv2.GaussianBlur(img,(5,5),1)
	# angle,aline,sline,clt = findwaistline(img,3,clt)

	rsz_fac = 128 / min(img.shape[0], img.shape[1])
	rsz_fac = 1
	img = cv2.resize(img, dsize=None, fx=rsz_fac, fy=rsz_fac)
	aline = [0, int(ly1[ctr]), img.shape[0], int(ly2[ctr])]
	minly1 = min(int(ly1[ctr]), int(ly2[ctr]))
	sline = [0, minly1, img.shape[0], minly1]
	angle = angles[ctr]
	img = frame
	if valid_angle:
		corner_pt = np.array([x1, y1])
		pt1 = tuple(aline[0:2] + corner_pt)
		pt2 = tuple(aline[2:] + corner_pt)
		cv2.arrowedLine(img, pt1, pt2, (0, 255, 255), 2, tipLength=0.05)
		pt1 = tuple(sline[0:2] + corner_pt)
		pt2 = tuple(sline[2:] + corner_pt)
		cv2.arrowedLine(img, pt1, pt2, (0, 255, 0), 2, tipLength=0.05)
		cv2.circle(img, pt1, 5, (0, 255, 0), cv2.FILLED)

	#
	# ly1.append(aline[1])
	# ly2.append(aline[3])
	# angles.append(angle)
	cv2.imwrite('outputs/' + str(counter) + '.png', img)

	# img = cv2.resize(img,(512,512))
	# cv2.imshow('Angle in Degrees: ' "{:.2f}".format(angle), img)
	cv2.imshow('vid', img)
	counter += 1
	writer2.write(img)
	cv2.waitKey(1)
#
writer2.release()
cv2.destroyWindow('vid')