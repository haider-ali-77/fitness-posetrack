import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from waistline_modules.color_quantization import cluster_quantization


# def imshow_components(labels):
# 	# Map component labels to hue val
# 	label_hue = np.uint8(179 * labels / np.max(labels))
# 	blank_ch = 255 * np.ones_like(label_hue)
# 	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
# 	# cvt to BGR for display
# 	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
# 	# set bg label to black
# 	labeled_img[label_hue == 0] = 0
# 	# labeled_img = cv2.resize(labeled_img,dsize=None,fx=2,fy=2)
# 	labeled_img = cv2.resize(labeled_img, dsize=None, fx=rsz_fac, fy=rsz_fac)
#
# 	# cv2.imshow('labeled.png', labeled_img)
# 	# cv2.waitKey()
# 	#
# 	return labeled_img
plot = False
def findwaistline(img,num_colors =2 ,clt = None):
	kernel = np.array((img.shape[0],img.shape[1]))
	kernel = (kernel*0.35).astype(np.int)
	if kernel[0]%2==0:
		kernel[0]+=1
	if kernel[1]%2==0:
		kernel[1]+=1
	img = cv2.GaussianBlur(img,tuple(kernel),0)


	cimg,clt = cluster_quantization(img,num_colors,clt)
	if plot:
		plt.imshow(img)
		plt.title('img')

		plt.show()
		plt.imshow(cimg)
		plt.title('cimg')
		plt.show()

	hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)

	# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
	h = hsv[:, :, 0]
	if np.max(h)==np.min(h):
		h=h.astype(np.float32)
	else:
		h = (h.astype(np.float32) - np.min(h))/(np.max(h)-np.min(h))
	h = h*255



	h = h.astype(np.uint8)
	# nlabels,labels,stats,cent = cv2.connectedComponentsWithStats(h)
	#
	# area = stats[:,cv2.CC_STAT_AREA]
	# lbls_order = np.argsort(area)
	# lbls_index = np.arange(0,nlabels)
	# sorted_lbls_index = lbls_index[lbls_order]
	# imh,imw = h.shape
	# mask = np.zeros((imh+2,imw+2),np.uint8)
	# for i in sorted_lbls_index[-2:]:
	# 	seedpoint = np.argwhere(labels ==i)[0]
	# 	cv2.floodFill(h,mask,seedpoint,labels[seedpoint])
	if plot:
		plt.imshow(h)
		plt.title('h')
		plt.show()

	# icolor = int(np.average(h[-1,:]))
	# thh=np.zeros((h.shape[0],h.shape[1],3),h.dtype)
	# thh[(h>icolor-20) & (h<icolor+20)]=255
	# img = cv2.resize(img,dsize=None,fx=rsz_fac,fy=rsz_fac)
	# h[h<200]=0
	h = cv2.Canny(h,0,50)

	if plot:
		plt.imshow(h)
		plt.title('Canny')
		plt.show()
	# num_lables,labels_im = cv2.connectedComponents(h)

	# labels_im = imshow_components(labels_im)
	# thresh = int(img.shape[0]*0.8)
	# lines = cv2.HoughLines(h,1,np.pi/180,3)
	#
	# for rho,theta in lines[0]:
	# 		a = np.cos(theta)
	# 		b = np.sin(theta)
	# 		x0 = a*rho
	# 		y0 = b*rho
	# 		x1 = int(x0 + 1000*(-b))
	# 		y1 = int(y0 + 1000*(a))
	# 		x2 = int(x0 - 1000*(-b))
	# 		y2 = int(y0 - 1000*(a))
	# 		color = np.random.randint(0,255,3)
	# 		color = [int(color[i]) for i in range(3)]
	# 		cv2.line(img,(x1,y1),(x2,y2),color,1)
	# 		# cv2.line(h,(x1,y1),(x2,y2),(0,0,255),1)

	h10p = int(h.shape[1] * 0.1) + 1
	h90p = int(h.shape[1] * 0.9) - 1
	h10 = h[:, :h10p]
	h90 = h[:, h90p:]
	h10 = np.average(h10, axis=1) > 0
	h90 = np.average(h90, axis=1) > 0
	mid = int(h.shape[1] *0.5)
	y1 = np.argwhere(h10[mid:])

	if len(y1) == 0:
		y1 = img.shape[0]
	else:
		if len(y1==1):

			y1 = y1[0, 0]
		else:
			y1 = y1[1,0]
		y1 += mid
	y2 = np.argwhere(h90[mid:])
	if len(y2) == 0:
		y2 = img.shape[0]
	else:
		if len(y2==1):

			y2 = y2[0, 0]
		else:
			y2 = y2[1,0]
		y2 += mid
	if y1==img.shape[0] and y2==img.shape[0]:
		ho = h[mid:,:]
		ho = np.average(ho,axis=1)>0
		y1 = np.argwhere(ho)

		if len(y1) == 0:
			y1 = img.shape[0]
		else:
			if len(y1==1):

				y1 = y1[0, 0]
			else:
				y1 = y1[1,0]
			y1 += mid
		y2 = y1

	elif y1==img.shape[0] or y2==img.shape[0]:
		if y2 > y1:
			y2 = y1
		else:
			y1 = y2
	x1= 0
	x2 = img.shape[1]
	angle =- np.arctan2(y2-y1,x2-x1)*180/np.pi
	waist_line = [x1,y1,x2,y2]
	straight_line = [x1,y1,x2,y1]
	return angle,waist_line,straight_line,clt

prev_waistline_params = None

def draw_waistline_front_back(frame,thickness, pts, clt=None, angle_font_params = None):
	global prev_waistline_params
	ptsc = pts[[5,6,11,12]]
	x1,y1 = np.min(ptsc,axis=0)
	x2,y2 = np.max(ptsc,axis=0)
	ptx,pty = (x1,y1)
	x1 = int(x1 + (x2-x1)*0.2)
	x2 = int(x2 - (x2-x1)*0.2)

	y1 = int(y1+(y2-y1)*0.25)
	y2 = int(y2+(y2-y1)*0.2)
	if y2>frame.shape[0]:
		y2 = frame.shape[0]
	if x2>frame.shape[1]:
		x2 = frame.shape[1]
	if y1<0:
		y1 = 0
	if x1<0:
		x1=0
	imgc = frame[y1:y2,x1:x2]
	angle, aline, sline,clt = findwaistline(imgc,2,clt)
	pt1 = tuple(aline[0:2])
	pt2 = aline[2:].copy()
	pt2[0] = int(pt2[0]-(pt2[0]-pt1[0])*0.1)
	pt2 = tuple(pt2)
	if prev_waistline_params is  None:
		if abs(angle)<10:
			prev_waistline_params = (angle,aline,sline)
		else:
			return clt,angle

	pangle,paline,psline = prev_waistline_params
	pt1d =  np.abs(np.array(pt1)-paline[:2])
	pt2d =  np.abs(np.array(pt2)-paline[2:])
	if np.max([pt1d,pt2d])/np.max(imgc.shape) > 0.25 or abs(angle)>10:
		print(angle,np.max([pt1d,pt2d])/np.max(imgc.shape))
		angle,aline,sline = prev_waistline_params
		pt1 = tuple(aline[0:2])
		pt2 = aline[2:].copy()
		pt2[0] = int(pt2[0]-(pt2[0]-pt1[0])*0.1)
		pt2 = tuple(pt2)
	else:
		prev_waistline_params = (angle,aline,sline)






	cv2.line(imgc, pt1, pt2, (255, 255, 255),thickness)
	pt1 = tuple(sline[0:2])
	pt2 = tuple(sline[2:])
	# cv2.arrowedLine(imgc, pt1, pt2, (0, 255, 0), 5, tipLength=0.02)

	cv2.circle(imgc, pt1, 5, (0, 255, 0), cv2.FILLED)
	if plot:
		plt.imshow(imgc)
		plt.show()
	frame[y1:y2,x1:x2]=imgc

	if angle_font_params is not None:
			text_pt2 = int(y2 - (y2-pty)*0.2)
			cv2.putText(frame,'Pelvis Angle: %.1f'%angle,(ptx-10,text_pt2),angle_font_params[0],angle_font_params[1],
			            angle_font_params[2],angle_font_params[3])

	return clt,angle
keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle"]

keypoints_map = {i: j for j, i in enumerate(keypoints)}
def draw_waistline_side(frame,frame2, pts, clt=None, angle_font_params = None):
	pts = pts.reshape(-1, 2)

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

	crop_orignal = frame[y1:y2, x1:x2]
	crop_processed = frame2[y1:y2, x1:x2]

	img = crop_orignal
	img_p = crop_processed

	# img  = cv2.GaussianBlur(img,(5,5),1)
	angle, aline, sline, clt = findwaistline(img, 2, clt)
	# if abs(angle)>30:
	# 	return clt,angle
	pt1 = tuple(aline[0:2])
	pt2 = aline[2:].copy()
	pt2[0] = int(pt2[0]-(pt2[0]-pt1[0])*0.1)
	pt2 = tuple(pt2)
	cv2.line(img_p, pt1, pt2, (255, 255, 255),5)#, tipLength=0.05)
	pt1 = tuple(sline[0:2])
	pt2 = tuple(sline[2:])
	# cv2.arrowedLine(imgc, pt1, pt2, (0, 255, 0), 5, tipLength=0.02)

	cv2.circle(img, pt1, 5, (0, 255, 0), cv2.FILLED)
	# plt.imshow(imgc)
	# plt.show()
	#frame2[y1:y2,x1:x2]=img_p

	if angle_font_params is not None:
		text_pt2 = int(y2 - (y2-y1)*0.2)
		cv2.putText(frame2,'Pelvis Angle: %.1f'%angle,(x1-10,text_pt2),angle_font_params[0],angle_font_params[1],
		            angle_font_params[2],angle_font_params[3])

	return clt,angle

if __name__ == '__main__':

	images_path = os.listdir('images/sample')
	np.random.shuffle(images_path)

	for img_path in images_path:
		if not img_path.endswith('.png'): continue
		img = cv2.imread('images/sample/' + img_path)
		angle,aline,sline = findwaistline(img)
		rsz_fac = 512 / min(img.shape[0], img.shape[1])
		img = cv2.resize(img,dsize=None,fx=rsz_fac,fy=rsz_fac)
		aline = [int(pt*rsz_fac) for pt in aline]
		sline = [int(pt*rsz_fac) for pt in sline]

		pt1 = tuple(aline[0:2])
		pt2 = tuple(aline[2:])
		cv2.arrowedLine(img,pt1,pt2,(0,255,255),2,tipLength=0.02)
		pt1 = tuple(sline[0:2])
		pt2 = tuple(sline[2:])
		cv2.arrowedLine(img,pt1,pt2,(0,255,0),2,tipLength=0.02)
		cv2.circle(img,pt1,5,(0,255,0),cv2.FILLED)


		# cv2.imwrite(img_path,img)

		# img = cv2.resize(img,(512,512))
		cv2.imshow('Angle in Degrees: ' "{:.2f}".format(angle), img)
		cv2.waitKey(0)
