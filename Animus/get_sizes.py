import cv2
import os

windows_dir = 'C:\\Users\\marce\\Documents\\osfstorage\\gold standard photos\\gold_standard_photos'
dir_of_interest = windows_dir

sizes = set()
for filename in os.listdir(dir_of_interest):
	if filename.endswith('jpg'):
		try:
			print('\\'.join([dir_of_interest, filename]))
			sizes.add(cv2.imread('\\'.join([dir_of_interest, filename])).shape)
		except AttributeError as e:
			print(filename)
			raise e
print(sizes)