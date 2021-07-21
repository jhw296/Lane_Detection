from time import sleep

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

line_image = 0

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


image = cv2.VideoCapture('solidWhiteRight.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
ret, frame = image.read()
height, width, layers = frame.shape
size = (width, height)

out = cv2.VideoWriter('video_out.mp4', fourcc, 20, size, 0)

region_of_interest_vertices = [(0, int(height)), (int(width)/2, int(height)/2), (int(width), int(height)),]



while True:
	ret, frame = image.read()
	if not ret:
		break
	gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	cannyed_image = cv2.Canny(gray_image, 100, 200)
	cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

	lines = cv2.HoughLinesP (cropped_image, rho = 6, theta = np.pi/60, threshold = 160, lines = np.array([]), minLineLength = 40, maxLineGap = 25 )
	#print(lines)

	#out.write(cropped_image)
	
	line_image = draw_lines(frame, lines)

	cv2.imshow('video', frame)
	cv2.imshow('video_gray', cropped_image)
	cv2.imshow('video_line', line_image)
	cv2.waitKey(1)
	sleep(0.1)
#image.release()
#out.release()
cv2.destroyAllWindows()

