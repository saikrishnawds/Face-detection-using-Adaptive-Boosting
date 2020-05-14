import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

# Class for Visualization
class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None
		self.chosen_filters=[]
		self.subset_str = "real"
	def draw_filters(self):
		drawings = []


		for filter in self.chosen_filters:
			draw = np.zeros((16,16))
			pos_rects,neg_rects = filter
			for x1,y1,x2,y2 in pos_rects:
				draw[int(x1):int(x2), int(y1):int(y2)] = 1
			for x1,y1,x2,y2 in neg_rects:
				draw[int(x1):int(x2), int(y1):int(y2)] = -1
			drawings.append(draw)

		self.plotImageSet(drawings,"Top Haar Filters")

	def plotImageSet(self, images, suptitle, cols=4, titles=None, scaledwn=4):

		assert ((titles is None) or (len(images) == len(titles)))
		n_images = len(images)
		if titles is None: titles = ['Filter (%d)' % i for i in range(n_images)]
		fig = plt.figure()

		for n, (image, title) in enumerate(zip(images, titles)):
			a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
			if image.ndim == 2:
				plt.gray()
			plt.imshow(image)
		
		fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scaledwn)
		fig.suptitle(suptitle)
		fig.savefig(suptitle + self.subset_str + '.png', bbox_inches='tight')

	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Weak Classifier Accuracies')

	def draw_sc_errors(self,cumsum):
		plt.figure()
		x=[]
		y=[]
		for t in range(len(cumsum)):
			x.append(t)
			y.append(np.mean(np.where(np.sign(cumsum[t]) != self.labels, 1, 0)))
		plt.plot(x, y)
		plt.ylabel('Error')
		plt.xlabel('Strong Classifiers')
		plt.title('Strong Classifier Errors')
		plt.legend(loc='upper right')
		plt.savefig('Strong Classifier Errors')
if __name__ == '__main__':
	main()
