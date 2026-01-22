import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from color_quantization import cluster_quantization


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

def findwaistline(img,num_colors =2 ,clt = None):
	cimg,clt = cluster_quantization(img,num_colors,clt)
	# plt.imshow(cimg)
	# plt.show()

	hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)

	# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
	h = hsv[:, :, 0] * (255 / 179)


	h = h.astype(np.uint8)
	nlabels,labels,stats,cent = cv2.connectedComponentsWithStats(h)
	# plt.imshow(h)
	# plt.show()
	# area = stats[:,cv2.CC_STAT_AREA]
	# lbls_order = np.argsort(area)
	# lbls_index = np.arange(0,nlabels)
	# sorted_lbls_index = lbls_index[lbls_order]
	# imh,imw = h.shape
	# mask = np.zeros((imh+2,imw+2),np.uint8)
	# for i in sorted_lbls_index[-2:]:
	# 	seedpoint = np.argwhere(labels ==i)[0]
	# 	cv2.floodFill(h,mask,seedpoint,labels[seedpoint])



	# icolor = int(np.average(h[-1,:]))
	# thh=np.zeros((h.shape[0],h.shape[1],3),h.dtype)
	# thh[(h>icolor-20) & (h<icolor+20)]=255
	# img = cv2.resize(img,dsize=None,fx=rsz_fac,fy=rsz_fac)

	h = cv2.Canny(h, 25, 50)
	# plt.imshow(h)
	# plt.show()
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
