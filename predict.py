import tensorflow as tf
slim = tf.contrib.slim
import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from os import listdir
from os.path import isfile, join
from os import walk
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve
from matplotlib import pyplot


#Uncomment to run on CPU, comment to run on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '' 

#Inception V1 uses images of 224x224x3
image_size = 224
session = tf.Session()

def get_test_images(mypath):
	"""
	Returns path to test images given path mypath 
	"""

	return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find('.jpg') != -1]

def transform_img_fn(path_list):
	""" 
	Transforms images from argumnt path_list
	"""

	out = []
	for f in path_list:
		image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
		image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
		out.append(image)
	return session.run([out])[0]

def fileToLabelDict(labels_dir):
	"""
	Converts text file into dict
	"""

	label_int_dict = {}
	with open(labels_dir) as f:
		lines = f.readlines()
	for line in lines:
		entry = line.rstrip().split(":")
		label_int_dict[entry[1]] = entry[0]
	return label_int_dict

def getTrueLabels(path_dir, labels_dir, key_to_label):
	""" 
	Given a directory path to images, makes a list of the correct labels via a labels file 
	(which we output when dividing the train and test set)

	Arguments:

	path_dir: 

	Path to file of images where we want the true labels

	Labels_to_int:

	Path to text file mapping labels --> int

	key_to_label:

	A dict mapping NEGATIVE, DELIVERED, SAB, BIOCHEMICAL --> labels 

	Returns:

	nparray of true labels 
	"""

	image_list = get_test_images(path_dir)
	true_labels = np.empty((len(image_list),))
	label_int_dict = fileToLabelDict(labels_dir)

	#Builds true labels 
	index = 0
	for image in image_list:
		if "NEGATIVE" in image:
			true_labels[index] = int(label_int_dict[key_to_label["NEGATIVE"]])
		if "DELIVERED" in image:
			true_labels[index] = int(label_int_dict[key_to_label["DELIVERED"]])
		if "SAB" in image:
			true_labels[index] = int(label_int_dict[key_to_label["SAB"]])
		if "BIOCHEMICAL" in image:
			true_labels[index] = int(label_int_dict[key_to_label["BIOCHEMICAL"]])
		index += 1
	return true_labels




def predict(train_dir, test_dir, num_classes):
	""" 
	Loads weights from a TF model and makes predictions.

	Arguments:

	train_dir: directory of trained model

	test_dir: directory of test images (split into folders by class)

	num_classes: number of classes of prediction

	Returns:

	Returns logits in the order of test images given and nparray of predictions
	"""

	processed_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

	with slim.arg_scope(inception.inception_v1_arg_scope()):
		logits, _ = inception.inception_v1(processed_images, num_classes=num_classes, is_training=False)

	probabilities = tf.nn.softmax(logits)

	def predict_fn(images):
		return session.run(probabilities, feed_dict={processed_images: images})

	#Loads in latest training checkpoint
	checkpoint_path = tf.train.latest_checkpoint(train_dir)
	init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,slim.get_variables_to_restore())
	init_fn(session)
	image_list = get_test_images(test_dir)
	images = transform_img_fn(image_list)
	predicted_probs = predict_fn(images)

	predictions = np.empty((len(predicted_probs),))

	for example in range(len(predicted_probs)):
		predictions[example] = np.argmax(predicted_probs[example,:])


	return predicted_probs, predictions


def plotROC(prediction_probs, actual):
	""" 
	Plots the ROC curve 

	Arguments:

	actual: ground truth labels

	prediction_probs: predicted probabilities 
	"""
	ns_probs = [0 for _ in range(len(actual))]
	lr_fpr, lr_tpr, _  = roc_curve(actual, prediction_probs)
	ns_fpr, ns_tpr, _ = roc_curve(actual, ns_probs)
	pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model')
	# axis labels
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	# show the plot
	pyplot.show()


def showStats(true_labels, predictions, labels_dir, showExtraStats, writeToFile=''):
	""" 
	Prints various statistics about model preformance.

	Arguments:

	true_labels: true labels of the predictions 

	predictions: predicted labels

	labels_dir: directory to text file of labels to integers

	Returns:

	N/A 
	"""

	#Print basic info
	print("Labels:")
	label_dict = fileToLabelDict(labels_dir)
	print(label_dict)
	print("\nPredictions:")
	print(predictions)
	print("\nActual:")
	print(true_labels)
	print("\n")

	#Builds confusion matrix and additional stats 
	my_inverted_dict = {}
	for elem in label_dict.keys():
		my_inverted_dict[elem] = int(label_dict[elem])
	my_inverted_dict = dict(zip(my_inverted_dict.values(), my_inverted_dict.keys()))
	cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=predictions)
	cm.relabel(mapping=my_inverted_dict)
	cm.print_matrix()
	if showExtraStats:
		#print("Micro F1 Score: ",cm.overall_stat['F1 Micro'])
		#print("Macro F1 Score: ",cm.overall_stat['F1 Macro'])
		#print("Cross Entropy: ",cm.overall_stat['Cross Entropy'])
		#print("95% CI: ",cm.overall_stat['95% CI'])
		print("AUC: ", cm.AUC)
		print("AUC quality:", cm.AUCI)
	#Outputs to output.txt
	if writeToFile == '':
		pass
	else:
		with open(writeToFile, 'w') as f:
			f.write("Labels:\n\n")
			f.write(str(label_dict))
			f.write("\n\nPredictions::\n\n")
			f.write(str(predictions))
			f.write("\n\nActual:\n\n")
			f.write(str(true_labels))
			f.write("\n\n")
			f.write(dictToString(cm.matrix))
			f.write("\n\n")
			f.write("AUC: \n\n")
			f.write(str(cm.AUC))
			f.write("AUC Quality: \n\n")
			f.write(str(cm.AUCI))
			f.write("\n\n")

def dictToString(dict1):
	""" 
	Writes a dictionary as a string to be outputted to a file
	in the form of a confusion matrix
	"""
	string = 'Predict    ' + '\t'
	for elem in dict1:
		string += elem +'\t'
	string += '\nActual\n\n'
	for elem in dict1:
		string += elem + '\t'
		for inner_elem in dict1[elem]:
			string += str(dict1[elem][inner_elem]) + '\t'
		string += '\n\n'
	return string




if __name__ == '__main__':
  """ 
  Converts the images in the train folder to TFrecords 
  """

  labels_dir = 'TrainedTFRecord/labels.txt'
  test_dir = 'Images_TwoClasses/test'
  #test_dir = 'Images_TwoClasses/train/Pregnant'
  labels_to_classes = {"NEGATIVE": "NotPregnant", "DELIVERED":"Pregnant", "SAB":"Pregnant", "BIOCHEMICAL":"NotPregnant"}
  true_labels = getTrueLabels(test_dir,labels_dir,labels_to_classes)
  write_file = 'Results/output.txt'

  train_dir = 'TrainedModel/'
  num_classes = 2
  prediction_logits, predictions = predict(train_dir, test_dir, num_classes)
  pred_prob = np.array(prediction_logits)[:,1]
  showStats(true_labels,predictions,labels_dir, True)
  plotROC(pred_prob, true_labels)
