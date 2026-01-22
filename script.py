import pickle
from densepose.data.structures import DensePoseResult
import cv2
import numpy as np
f = open('outputs/iuv.pkl', 'rb')
data = pickle.load(f)
img_id, instance_id = 0, 0  # Look at the first image and the first detected instance
# bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
areas = []
for j in range(0, len(data[0]['pred_densepose'])):
    bbox_xyxy = data[img_id]['pred_boxes_XYXY'][j]
    area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
    areas.append(area)
instance_id = np.argmax(areas)
bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
result_encoded = data[img_id]['pred_densepose'].results[instance_id]
iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
cv2.imwrite('outputs/pose0.png',iuv_arr[0,:,:])
cv2.imwrite('outputs/pose1.png',iuv_arr[1,:,:])
cv2.imwrite('outputs/pose2.png',iuv_arr[2,:,:])
