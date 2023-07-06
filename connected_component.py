import cv2
import numpy as np
import random

label = []
label_dict = {}

def getLabel(neighbours):

	if all(x==0 for x in neighbours):
		if len(label) == 0:
			label.append(1)
			return max(label)
		else:
			label.append(max(label) + 1)
			return max(label)
	else:
		max_label = 0
		min_label = 0

		neighbours = [x for x in neighbours if x != 0]
		neighbours.sort()

		min_label = neighbours[0]
		max_label = neighbours[len(neighbours)-1]

		if max_label == min_label:
			return min_label
		else:
			label_dict[max_label] = min_label
			return min_label

def connected_8(image_label):
	row, column = image_label.shape
	# First Pass
	for i in range(row):
		for j in range(column):

			# Only interested in pixels with value v [255] i.e white pixels
			if image_label[i,j] == 255:

				# Checking for different positions of pixels
				if i == 0 and j == 0:
					image_label[i,j] = getLabel([])

				elif i == 0 and j > 0:
					image_label[i,j] = getLabel([image_label[i,j-1]])

				elif i > 0 and j == 0:
					image_label[i,j] = getLabel([image_label[i-1,j], image_label[i-1,j+1]])

				elif i > 0 and j == (column-1):
					image_label[i,j] = getLabel([image_label[i-1,j-1], image_label[i-1,j], image_label[i,j-1]])

				elif i > 0 and j > 0:
					image_label[i,j] = getLabel([image_label[i-1,j-1], image_label[i-1,j], image_label[i-1,j+1], image_label[i,j-1]])
	# Second Pass
	for k in range(len(label_dict)):
		for i in range(row):
			for j in range(column):
				if image_label[i][j] in label_dict:
					image_label[i][j] = label_dict[image_label[i][j]]
	return image_label


def connected_4(image_label):
	row, column = image_label.shape
	# First Pass
	for i in range(row):
		for j in range(column):

			# Only interested in pixels with value v [255] i.e white pixels
			if image_label[i,j] == 255:

				# Checking for different positions of pixels
				if i == 0 and j == 0:
					image_label[i,j] = getLabel([])

				elif i == 0 and j > 0:
					image_label[i,j] = getLabel([image_label[i,j-1]])

				elif i > 0 and j == 0:
					image_label[i,j] = getLabel([image_label[i-1,j]])

				# elif i > 0 and j == (column-1):
				# 	image_label[i,j] = getLabel([image_label[i-1,j], image_label[i,j-1]])

				elif i > 0 and j > 0:
					image_label[i,j] = getLabel([image_label[i-1,j], image_label[i,j-1]])
	# Second Pass
	for k in range(len(label_dict)):
		for i in range(row):
			for j in range(column):
				if image_label[i][j] in label_dict:
					image_label[i][j] = label_dict[image_label[i][j]]
	return image_label


img = cv2.imread('flower2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
B = np.array([[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]], dtype = np.uint8)
threshold = cv2.dilate(threshold.copy(), B, iterations=1)
row, column = threshold.shape
new_img = np.array(threshold)
new_img = connected_8(new_img)

# Colorizing the labels
output_img = np.zeros((row, column, 3), np.uint8)
labelColor = {0: (0, 0, 0)}
count  = []
for i in range(row):
	for j in range(column):
		label = new_img[i,j]
		if label not in count and label not in labelColor:
			count.append(label)
		if label not in labelColor:
			labelColor[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
		output_img[i, j, :] = labelColor[label]

print(len(count))
cv2.imwrite("new_8.png", output_img)
# cv2.imshow('test', output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()