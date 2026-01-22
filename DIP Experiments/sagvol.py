import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy
cap = cv2.VideoCapture('trisha_right_20ft_slomo_IMG_2495.mov')
# writer = cv2.VideoWriter('original.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
writer2 = cv2.VideoWriter('sagvol.avi',cv2.VideoWriter_fourcc('D','I','V','X'),fps,(w,h))

#%%

ptsv = np.load('trisha_right.npy')
ptsv = ptsv[:,:,:2]
ptsv_raw = ptsv.copy()
hip_ptsv = ptsv[:,11:13,:]
plt.plot(hip_ptsv[:,0,1])

plt.plot(hip_ptsv[:,1,1])
plt.title('ptsv raw')
plt.show()

ptsv = savgol_filter(ptsv.reshape(-1,34),polyorder=3,window_length=21,axis=0).reshape(-1,17,2)

hip_ptsv = ptsv[:,11:13,:]
def get_hip_avg_kernel_size(sig):
	pk1 = scipy.signal.find_peaks(sig)[0]
	# plt.scatter(x=pk1,y=hip_ptsv[pk1,0,1])
	delta_pk1 = np.abs(np.diff(sig[pk1],append=0))
	delta_pk1[-1]=0

	n1 = np.argmax(delta_pk1)
	return int((pk1[n1+1]-pk1[n1])/3)

plt.plot(hip_ptsv[:,0,1])
plt.plot(hip_ptsv[:,1,1])
plt.title('sagvol small')
plt.show()




hip_ptsv_savgol  = hip_ptsv.copy()
# print(pk1)
kernel_size = get_hip_avg_kernel_size(hip_ptsv[:,0,1])+get_hip_avg_kernel_size(hip_ptsv[:,1,1])
kernel_size =int(kernel_size/2)
print(kernel_size)
avg_kernel = np.ones(kernel_size)/kernel_size
hip_ptsv[:,0,1] = np.convolve(hip_ptsv[:,0,1],avg_kernel,'same')
hip_ptsv[:,1,1] = np.convolve(hip_ptsv[:,1,1],avg_kernel,'same')
hip_ptsv[:kernel_size] = hip_ptsv_savgol[:kernel_size]
hip_ptsv[-kernel_size:] = hip_ptsv_savgol[-kernel_size:]
plt.plot(hip_ptsv[:,0,1])
plt.plot(hip_ptsv[:,1,1])
plt.title('avg')
plt.show()


ptsv[:,11:13,:] = hip_ptsv

exit(0)
#%%

#
# ptsv = np.load('trisha_right.npy')
# ptsv = ptsv[:,:,:2]
for frame_ctr,pts in enumerate(ptsv):
	pts = pts.astype(np.int)
	ret,frame = cap.read()

	for i,pt in enumerate(pts):

		if i == 11 or i==12:
			pt_raw = ptsv_raw[frame_ctr,i,:]
			cv2.circle(frame,tuple(pt),4,(0,0,255),cv2.FILLED)

			cv2.circle(frame,tuple(pt),5,(0,255,255),cv2.FILLED)


		cv2.circle(frame,tuple(pt),3,(0,255,0),cv2.FILLED)
	writer2.write(frame)

	frame = cv2.resize(frame,dsize=None,fx=0.7,fy=0.7)
	cv2.imshow("vid",frame)
	cv2.waitKey(int(1000/60))
cv2.destroyWindow("vid")
writer2.release()