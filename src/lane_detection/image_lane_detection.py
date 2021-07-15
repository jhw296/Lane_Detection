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
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8,)
	
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
	img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
	return img

region_of_interest_vertices = [(0, 540), (960/2, 540/2), (960, 540),]

image = mpimg.imread('solidWhiteCurve.jpg')

plt.figure()
plt.imshow(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

plt.figure()
plt.imshow(cropped_image)

lines = cv2.HoughLinesP (cropped_image, rho = 6, theta = np.pi/60, threshold = 160, lines = np.array([]), minLineLength = 40, maxLineGap = 25 )

line_image = draw_lines(image, lines)
plt.figure()
plt.imshow(line_image)
plt.show()
