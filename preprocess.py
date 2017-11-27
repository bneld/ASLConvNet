import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import imageio

# hand5_e_bot_seg_5_cropped.png
class image() : 
	def __init__(self, signer_num, gesture , ill , R, matrix):
		self.signer_num  = signer_num
		self.label = gesture
		self.label_vec = np.zeros(36)
		order = ord(gesture)
		if order <= 57: 
			self.label_vec[order - 48] = 1 

		else : 
			self.label_vec[order - 87] = 1

		self.ill = ill
		self.R = R
		self.matrix = matrix

def create_imageset():
	#Variables
	images_path = './images'

	#Read for images folder 
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	#processing and reading image files
	max_width =0 
	max_height = 0 
	min_width = 999999999999999
	min_height = 999999999999999999
	image_set = [] #contains all images

	for i in image_files : 
		#split 
		info = i.split('_')
		# print(info)
		matrix = imageio.imread(images_path + '/' + i)
		
		if matrix.shape[0] > max_height : 
			max_height = matrix.shape[0]

		if matrix.shape[0] < min_height : 
			min_height = matrix.shape[0]

		if matrix.shape[1] > max_width : 
			max_width = matrix.shape[1]

		if matrix.shape[1] < min_width : 
			min_width = matrix.shape[1]

		image_set.append(image(int(info[len(info[0])-1]), info[1] , info[2] , info[4] , matrix))
		


	dataset_info = [min_height, max_height , min_width, max_width]
	final_result = [image_set , dataset_info]

	return final_result



