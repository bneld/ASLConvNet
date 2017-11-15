import cv2

# hand5_e_bot_seg_5_cropped.png
class image() : 
	def __init__(self, signer_num, gesture , ill , R):
		self.signer_num  = signer_num
		self.g = gesture 
		self.ill = ill
		self.R = R


#Read 