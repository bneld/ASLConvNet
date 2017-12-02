import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import imageio

#for padding purposes 
max_width  = 28 
max_height = 28

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

def scale_image(image, dim):
	# needs dimension = (width, height)
	return cv2.resize(image, (dim[1], dim[0]))
def scale_height_keep_ratio(image, target_height):
	r = target_height / image.shape[0]
	return (target_height, int(image.shape[1] * r))
def scale_width_keep_ratio(image, target_width):
	r = target_width / image.shape[1]
	return (int(image.shape[0] * r), target_width)

def pad_image(target_image) : 
	#get current dimensions
	current_width = target_image.shape[1] 
	current_height =  target_image.shape[0]

	#calc how much padding needed 
	pad_height = max_height - current_height 
	pad_width = max_width  - current_width

	pad_top, pad_bottom, pad_left, pad_right = 0,0,0,0
	if pad_height % 2 == 1:
		# if difference odd, assign accordingly
		pad_top = pad_height // 2
		pad_bottom = pad_height // 2 + 1
	else:
		pad_top, pad_bottom = pad_height / 2 ,  pad_height / 2

	if pad_width % 2 == 1:
		pad_right = pad_width // 2
		pad_left = pad_width // 2 + 1
	else:
		pad_right, pad_left = pad_width / 2 , pad_width / 2

	return cv2.copyMakeBorder(target_image,int(pad_top),int(pad_bottom),int(pad_left),int(pad_right),cv2.BORDER_CONSTANT)

def create_imageset(excludeHardClasses=True):
	#Variables
	images_path = './images'

	#Read for images folder 
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	#processing and reading image files
	image_set = [] #contains all images

	for i in image_files : 
		#split 
		info = i.split('_')
		matrix = cv2.imread(images_path + '/' + i)
		RGB_img = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
		# scale image
		height, width = RGB_img.shape[0], RGB_img.shape[1]
		dimensions = scale_height_keep_ratio(RGB_img, max_height) if height > width else scale_width_keep_ratio(RGB_img, max_width)
		RGB_img = scale_image(RGB_img, dimensions)
		# pad image to 100 x 100
		matrix = pad_image(RGB_img)
		# cv2.imshow('image',matrix)

		
		image_set.append(image(int(info[len(info[0])-1]), info[1] , info[2] , info[4] , matrix))

		# plt.subplot(231),plt.imshow(RGB_img,'gray'),plt.title('ORIGINAL')
		# plt.subplot(236),plt.imshow(matrix,'gray'),plt.title('NEW')
		# plt.show()

	return image_set


