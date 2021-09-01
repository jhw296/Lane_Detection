from time import sleep

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

line_image = 0
prev_poly_left = 0
prev_poly_right = 0

def region_of_interest (img, vertices) :
	mask = np.zeros_like(img)
	match_mask_color = 255
	cv2.fillPoly (mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and (img, mask)
	return masked_image

def draw_lines(img, lines, color = [255, 0, 0], thickness=3):
	if lines is None:
		return
	img = np.copy(img)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)

	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
	img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
	return img

# image = cv2.VideoCapture('solidWhiteRight.mp4')
image = cv2.VideoCapture('video4.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')



out = cv2.VideoWriter('video_out.mp4', fourcc, 20, size, 0)

region_of_interest_vertices = [(0, int(height)), (int(width)/2, int(height)/2), (int(width), int(height)),]

def pipeline(image):
	#here
	left_line_x = []
	left_line_y = []
	right_line_x = []
	right_line_y = []
	global prev_poly_left
	global prev_poly_right

	# min_y = int(image.shape[0] * (3 / 5))
	min_y = 320
	max_y = int(image.shape[0])

	for line in lines:
		for x1, y1, x2, y2 in line:
			# print(x1, y1, x2, y2)
			slope = float((y2 - y1)) / float((x2 - x1))  # fix error (TypeError)
			# print(slope)
			if math.fabs(slope) < 0.5:
				continue
			if slope <= 0:
				left_line_x.extend([x1, x2])
				left_line_y.extend([y1, y2])
			else:
				right_line_x.extend([x1, x2])
				right_line_y.extend([y1, y2])


	# print(line)
	# print(left_line_x[0], left_line_x[1])
	# print(left_line_x, left_line_y, right_line_x, right_line_y)

	if left_line_x is not None and left_line_y is not None:
		try:
			poly_left = np.poly1d(np.polyfit(
				left_line_y,
				left_line_x,
				deg=1
			))
			prev_poly_left = poly_left
		except:
			poly_left = prev_poly_left

		if poly_left is not None:
			left_x_start = int(poly_left(max_y))
			left_x_end = int(poly_left(min_y))
	if left_line_x is not None and right_line_y is None:
		pass

	if right_line_x is not None and right_line_y is not None:
		try:
			poly_right = np.poly1d(np.polyfit(
				right_line_y,
				right_line_x,
				deg=1  # TypeError: expected non-empty vector for x (x = NX.asarray(x) + 0.0, x.size == 0)
			))
			prev_poly_right = poly_right
		except:
			poly_right = prev_poly_right

		if poly_right is not None:
			right_x_start = int(poly_right(max_y))
			right_x_end = int(poly_right(min_y))
	if right_line_x is not None and right_line_y is not None:
		pass


	line_image = draw_lines(
		image,
		[[
			[int(left_x_start), int(max_y), int(left_x_end), min_y],
			[int(right_x_start), int(max_y), int(right_x_end), min_y],
		]],
		#	thickness=5
		(0, 0, 255),
		3,
	)
	return line_image

while True:
	ret, frame = image.read()
	if not ret:
		break
	gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	cannyed_image = cv2.Canny(gray_image, 100, 200)
	cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

	lines = cv2.HoughLinesP (cropped_image, rho = 6, theta = np.pi/60, threshold = 160, lines = np.array([]), minLineLength = 40, maxLineGap = 25 )

	# print(lines)

	#out.write(cropped_image)

	#line_image = draw_lines(frame, lines)

	# cv2.imshow('video', frame)
	# cv2.imshow('video_gray', cropped_image)

	cv2.imshow('video_line', pipeline(frame))
	cv2.waitKey(1)
#image.release()
#out.release()
cv2.destroyAllWindows()

