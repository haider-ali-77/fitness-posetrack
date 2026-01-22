import asyncio
import requests
import asyncio
from timeit import default_timer
from concurrent.futures import ThreadPoolExecutor
import os
import matplotlib.pyplot as plt
import cv2

import boto3
import io
import base64


class FaceBlurring:
	def __init__(self):
		self.client = boto3.client('rekognition')


	def detect_faces_local(self, photo):#, client):
		# #client = boto3.client('rekognition')

		with open(photo, 'rb') as image:
			response = self.client.detect_faces(Image={'Bytes': image.read()})
		return response

	def face_blur_main(self, in_folder, i, out_folder, ext,frames):#, client):
		# for i in range(t_images):
		if int(i)>frames:
			return
		else:
			name = '%i.PNG' % (i + 1)
			# photo = os.path.join(in_folder, name)
			photo = os.path.join(in_folder, name)
			###
			img = plt.imread(photo)
			im_h = img.shape[0]
			im_w = img.shape[1]
			try:
				response = self.detect_faces_local(photo)#, client)
				num_faces = len(response['FaceDetails'])
				print('Face blurring ' ,i )

				if num_faces > 0:
					for j in range(num_faces):
						bbox = response['FaceDetails'][j]['BoundingBox']

						left_c = int(bbox['Left'] * im_w)
						top_c = int(bbox['Top'] * im_h)
						width_c = int(bbox['Width'] * im_w)
						height_c = int(bbox['Height'] * im_h)

						crp = img[top_c:top_c + height_c, left_c:left_c + width_c]
						crp = cv2.blur(crp, (int(width_c / 2), int(width_c / 2)))
						img[top_c:top_c + height_c, left_c:left_c + width_c] = crp

				# cv2.rectangle(img, (10,10), (100,100), (0,0,255), 4)
			except:
				pass
			# plt.imsave(os.path.join(out_folder, name), img)
			im_save = '%i.%s'%(i + 1, ext)
			plt.imsave(os.path.join(out_folder, im_save), img)



	async def face_blur_async(self, in_folder, out_folder, ext,frame):#, client):
		images = os.listdir(in_folder)
		t_images = len(images)
		print('Total images ',t_images)
	####
		with ThreadPoolExecutor(max_workers=8) as executor:
			loop = asyncio.get_event_loop()

			tasks = [loop.run_in_executor(executor, self.face_blur_main,*(in_folder, i, out_folder, ext,frame))for i in range(t_images)]
			for response in await asyncio.gather(*tasks):
				pass


	def start_face_blurring(self, in_folder, out_folder, ext,frame):
		loop = asyncio.get_event_loop()
		future = asyncio.ensure_future(self.face_blur_async(in_folder, out_folder, ext,frame))
		loop.run_until_complete(future)

	def detect_face(self,image_name):
		img=cv2.imread(image_name)
		im_h = img.shape[0]
		im_w = img.shape[1]
		respons=self.detect_faces_local(image_name)
		num_faces = len(respons['FaceDetails'])
		print(image_name)
		faces = []
		if num_faces > 0:
			for j in range(num_faces):
				bbox = respons['FaceDetails'][j]['BoundingBox']
				left_c = int(bbox['Left'] * im_w)
				top_c = int(bbox['Top'] * im_h)
				width_c = int(bbox['Width'] * im_w)
				height_c = int(bbox['Height'] * im_h)
				faces.append(left_c)
				faces.append(top_c)
				faces.append(width_c)
				faces.append(height_c)

				crop = img[top_c:top_c + height_c, left_c:left_c + width_c]
		return faces
