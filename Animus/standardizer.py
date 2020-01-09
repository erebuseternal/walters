import cv2
import sys
import os
import numpy as np

def standardize(image, size):
	# 1. Crop out headers and footers
	image = crop_headers(image)
	# 2. Standardize ratio
	image = reshape_to_ratio(image, size)
	# 3. Shrink or expand
	image = cv2.resize(image, size[::-1])
	return image

def crop_headers(image):
	# start by chopping bottom and top ten pixels (to get rid
	# of shadows in the headers)
	image = image[10:-10,:,:]
	# now we do proper header chopping
	top_value_left = image[0:20, 0, :]
	top_change_point_left = min(
		np.where(image[:,0,c] != top_value_left[c])[0][0]
		for c in (0, 1, 2)
	)
	top_value_right = image[0, -1, :]
	top_change_point_right = min(
		np.where(image[:,-1,c] != top_value_right[c])[0][0]
		for c in (0, 1, 2)
	)
	top_change_point = max(top_change_point_right, top_change_point_left)
	bottom_value_left = image[-1, 0, :]
	bottom_change_point_left = max(
		np.where(image[:,0,c] != bottom_value_left[c])[0][-1]
		for c in (0, 1, 2)
	)
	bottom_value_right = image[-1, -1, :]
	bottom_change_point_right = max(
		np.where(image[:,-1,c] != bottom_value_right[c])[0][-1]
		for c in (0, 1, 2)
	)
	bottom_change_point = min(bottom_change_point_right, bottom_change_point_left)
	image = image[top_change_point:bottom_change_point+1,:,:]
	return image

def reshape_to_ratio(image, size):
	image_ratio = image.shape[0] / image.shape[1]
	size_ratio = size[0] / size[1]
	if size_ratio > image_ratio:
		coef = size_ratio / image_ratio
		new_x_size = coef * image.shape[0]
		x_padding = int((new_x_size - image.shape[0])/2) 
		image = cv2.copyMakeBorder(image, x_padding, x_padding, 0, 0, 
								   cv2.BORDER_CONSTANT)
	if image_ratio > size_ratio:
		coef = image_ratio / size_ratio 
		new_y_size = coef * image.shape[1]
		y_padding = int((new_y_size - image.shape[1])/2) 
		image = cv2.copyMakeBorder(image, 0, 0, y_padding, y_padding, 
								   cv2.BORDER_CONSTANT)
	return image


if __name__ == '__main__':
	windows_dir = (
		'C:\\Users\\marce\\Documents\\osfstorage'
		+ '\\gold standard photos\\gold_standard_photos'
	)
	x_size = int(sys.argv[1])
	y_size = int(sys.argv[2])
	i = 0
	for filename in os.listdir(windows_dir):
		if filename.endswith('jpg'):
			#print(filename)
			image = cv2.imread('\\'.join([windows_dir, filename]))
			image = standardize(image, (x_size, y_size))
			cv2.imwrite('\\'.join(['images', filename]), image)
			i += 1
			if i % 100 == 0:
				print(i)
