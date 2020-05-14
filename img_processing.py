import numpy as np
import cv2

from utils import integrate_images
from sklearn.feature_extraction import image as IMG

#Here I am extracting  patches from the image for all scales of the image, and returning the integrating images and the coordinates of the patches


def im2p(scales, image, p_w = 16, p_h = 16):
	all_pes = np.zeros((0, p_h, p_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		pes = IMG.extract_pes_2d(simage, (p_w, p_h)) # move along the row first

		total_p = pes.shape[0]
		row_p = (height - p_h + 1)
		col_p = (width - p_w + 1)
		assert(total_p == row_p * col_p)
		scale_xyxy = []
		for pid in range(total_p):
			y1 = pid / col_p
			x1 = pid % col_p
			y2 = y1 + p_h - 1
			x2 = x1 + p_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_pes = np.concatenate((all_pes, pes), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	return integrate_images(normalize(all_pes)), all_x1y1x2y2

#Here, I am returning  a vector of predictions (0/1) after nms, same length as scores, for which the i/p and o/p are as below:
#i/p: [x1, y1, x2, y2, score], threshold used for nms
#o/p: [x1, y1, x2, y2, score] after nms

# non- maxima suppression
    
def nms(xyxys, overlap_thresh):
	xyxys = xyxys
	# if there are no xyxys, return an empty list
	if len(xyxys) == 0:
		return []

	if xyxys.dtype.kind == "i":
		xyxys = xyxys.astype("float")
	xyxys = xyxys[xyxys[:, 4] >= 0.70]
	chosen = []

	# taking coordinates of the bounding boxes
	x1 = xyxys[:, 0]
	y1 = xyxys[:, 1]
	x2 = xyxys[:, 2]
	y2 = xyxys[:, 3]
	scores = xyxys[:, 4]
	# computing the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
	
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	indices = np.argsort(scores)
	while len(indices) > 0:
		last = len(indices) - 1
		i = indices[last]
		chosen.append(i)

		# finding the largest (x, y) coordinates for the start and the smallest (x, y) coordinates for the end of the bounding box 
	    
		xx1 = np.maximum(x1[i], x1[indices[:last]])
		yy1 = np.maximum(y1[i], y1[indices[:last]])
		xx2 = np.minimum(x2[i], x2[indices[:last]])
		yy2 = np.minimum(y2[i], y2[indices[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[indices[:last]]

		indices = np.delete(indices, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
	print(len(xyxys),len(xyxys[chosen]))
	return xyxys[chosen]

def normalize(images):
	standard = np.std(images)
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	pes, p_xyxy = im2p(scales, original_img)
	print(pes.shape)
	print(len(p_xyxy))
if __name__ == '__main__':
	main()
