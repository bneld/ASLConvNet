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
		self.ill = ill
		self.R = R
		self.matrix = matrix

#Variables
images_path = './images'

#Read for images folder 
image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

#processing and reading image files

image_set = [] #contains all images

for i in image_files : 
	#split 
	info = i.split('_')
	print(info)
	image_set.append(image(int(info[len(info[0])-1]), info[1] , info[2] , info[4] , imageio.imread(images_path + '/' + i)))
