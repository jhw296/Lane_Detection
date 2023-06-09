import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

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

# region_of_interest_vertices = [(0, 540), (960/2, 540/2), (960, 540),]
region_of_interest_vertices = [(0, 480), (640/2, 480/2), (640, 480),]

# image = mpimg.imread('solidWhiteCurve.jpg')
image = mpimg.imread('image.jpg')

# plt.figure()
# plt.imshow(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

# plt.figure()
# plt.imshow(cropped_image)

# HoughLines, HoughLinesP 
lines = cv2.HoughLinesP (cropped_image, rho = 6, theta = np.pi/60, threshold = 160, lines = np.array([]), minLineLength = 40, maxLineGap = 25)

print(len(lines))
print(lines)

# line_image = draw_lines(image, lines)
# plt.figure()
# plt.imshow(line_image)


left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []


for line in lines:
	for x1, y1, x2, y2 in line:
		print(x1, y1, x2, y2)
		slope = float((y2 - y1)) / float((x2 - x1)) # fix error (TypeError)
		print(slope)
		if math.fabs(slope) < 0.5:
			continue
		if slope <= 0:
			left_line_x.extend([x1, x2])
			left_line_y.extend([y1, y2])
		else:
			right_line_x.extend([x1, x2])
			right_line_y.extend([y1, y2])

# print(line)
# print(left_line_x, left_line_y, right_line_x, right_line_y)

min_y = 320
# min_y = image.shape[0]*(3/5)
max_y = image.shape[0]

poly_left = np.poly1d(np.polyfit(
	left_line_y,
	left_line_x,
	deg = 1
))

left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))

poly_right = np.poly1d(np.polyfit(
	right_line_y,
	right_line_x,
	deg = 1		# TypeError: expected non-empty vector for x (x = NX.asarray(x) + 0.0, x.size == 0)
))

right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))


# y-axis correction
line_image = draw_lines(
	image,
	[[
		[left_x_start, max_y, left_x_end, min_y],
		[right_x_start, max_y, right_x_end, min_y],
	]],
	thickness=5,
	# (0, 0, 255),
	# 3,
)

plt.figure()
plt.imshow(line_image)

plt.show()