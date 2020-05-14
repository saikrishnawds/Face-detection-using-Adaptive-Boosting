from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed


# Class for Weak Classifers:

class WC(ABC):
	# I am initializing a harr filter with the positive and negative rects which are in the form of [x1, y1, x2, y2] 0-index
	
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins

		self.activations = None

	# taking in one integrated img and return the value after applying the img
	# integrated_img is a 2D np array and return value is the number BEFORE polarity is applied
	
	
	def apply_filter2img(self, integrated_img):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_img[rect[3], rect[2]] \
				   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_img[rect[1] - 1, rect[0] - 1]) \
				   - (0 if rect[1] == 0 else integrated_img[rect[1] - 1, rect[2]]) \
				   - (0 if rect[0] == 0 else integrated_img[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_img[rect[3], rect[2]] \
				   + (0 if rect[1] == 0 or rect[0] == 0 else integrated_img[rect[1] - 1, rect[0] - 1]) \
				   - (0 if rect[1] == 0 else integrated_img[rect[1] - 1, rect[2]]) \
				   - (0 if rect[0] == 0 else integrated_img[rect[3], rect[0] - 1])
		return pos - neg

	# taking in a list of integrated imgs and calculate values for each img and integrated imgs are passed in as a 3-D np-array
	
	# calculating activations for all imgs before polarity is applied
	def apply_filter(self, integrated_imgs):
		values = []
		for idx in range(integrated_imgs.shape[0]):
			values.append(self.apply_filter2img(integrated_imgs[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying, activation length = %s' % (self.id + 1, len(values)))
		return values

	# calc_err is usedto compute the err of applying this weak classifier to the dataset given current weights
	
	@abstractmethod
	def calc_err(self, weights, labels):
		pass

	@abstractmethod
	def predict_img(self, integrated_img):
		pass


class Ada_WC(WC):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.sorted_indices = None
		self.polarity = None
		self.threshold = None

	def calc_err(self, weights, labels):
		# https://courses.cs.washington.edu/courses/cse576/17sp/notes/FaceDetection17.pdf
		n = weights.shape
		#errs = np.zeros(weights.shape[0])
		#polarities = np.zeros(weights.shape)
		new_weights = weights[self.sorted_indices]
		new_labels = labels[self.sorted_indices]
		pos_mask = new_labels==1
		neg_mask = new_labels==-1

		#print("new wts",new_weights.shape)
		FS = np.cumsum(np.where(pos_mask, new_weights,0))
		BG = np.cumsum(np.where(neg_mask, new_weights,0))

		AFS = FS[-1]
		ABG = BG[-1]

		right = FS + ABG - BG
		left = BG + AFS - FS
		errs=np.minimum(left,right)

		min_index 	   = np.argmin(errs)
		self.threshold = self.activations[self.sorted_indices[min_index]]
		self.polarity  = -1 if left[min_index] < right[min_index] else 1
		final_err    = errs[min_index]

		predictions = self.polarity * np.sign(self.activations - self.threshold)
		return final_err,predictions,self.threshold,self.polarity


	def predict_img(self, integrated_img):
		value = self.apply_filter2img(integrated_img)
		return self.polarity * np.sign(value - self.threshold)


class Real_WC(WC):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None  
		self.bin_pqs = None
		self.train_assignment = None
		self.sorted_activations = None
	def calc_err(self, weights, labels):
		n=weights.shape[0]
		max_act = max(self.activations)
		min_act = min(self.activations)
		step_size = (max_act-min_act)/self.num_bins
		p_b = np.zeros(self.num_bins)
		q_b = np.zeros(self.num_bins)
		bin_left=[(min_act+i*step_size) for i in range(0,self.num_bins)]
		bin_right=[left + step_size for left in bin_left]
		for bin in range(self.num_bins):
			is_in_bin = [True if act_i>bin_left[bin] and act_i<=bin_right[bin] else False for act_i in self.activations]
			is_in_bin = np.array(is_in_bin)
			p_b[bin] = sum(weights[np.logical_and(is_in_bin,labels== 1)])
			q_b[bin] = sum(weights[np.logical_and(is_in_bin,labels==-1)])
		loss = 2*np.sum(np.sqrt(p_b*q_b))
		p_b[p_b == 0] = 1e-7
		q_b[q_b == 0] = 1e-7

		self.bin_pqs=np.zeros((2,self.num_bins))
		self.bin_pqs[0] = p_b
		self.bin_pqs[1] = q_b
		predict=[]
		self.thresholds = np.array(bin_right[:self.num_bins-1])
		for i in range(n):
			bin_idx = np.sum(self.thresholds < self.activations[i])
			#print("bin_idx",bin_idx)
			predict.append(0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx]))
		return loss,predict,self.thresholds,self.bin_pqs

	def predict_img(self, integrated_img):
		value = self.apply_filter2img(integrated_img)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])


def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	ada_hf = Ada_WC(1, plus_rects, minus_rects, num_bins)
	real_hf = Real_WC(2, plus_rects, minus_rects, num_bins)


if __name__ == '__main__':
	main()
