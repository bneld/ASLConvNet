import preprocess
#import alex_net2.py


#images result
result = preprocess.create_imageset() 
# print(result[1])



#images result
print('\n\n\n***************')

for i in range(400,450):
	print(result[0][i].label)
	print(result[0][i].label_vec)
	print('\n****')

#How 324 x 324 

