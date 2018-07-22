import random
import csv
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Python script for confusion matrix creation.

def load_csv(filename):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset=list(lines)
#    print(dataset)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column])
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
   # print(lookup)
   #print(dataset)
    return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Initialize a network
def initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs):
        #print("No. of  Nuerons in Input layer: ", n_inputs)
        #print("No. of  Nuerons in Hidden Layer 1: ", n_hidden1)
        #print("No. of  Nuerons in Hidden Layer 2: ", n_hidden2)
        #print("No. of  Nuerons in Output layer: ", n_outputs)
        network = list()
        hidden_layer1 = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer1)
        hidden_layer2 = [{'weights':[random() for i in range(n_hidden1 + 1)]} for i in range(n_hidden2)]
        network.append(hidden_layer2)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
       	#print(network)
        return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]       #Complete the missing statement
	return activation


#Complete the missing function to Transfer neuron activation

def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))





# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        #For Hidden layer
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
		#For Output layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate): # introduced bug
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

 # Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def applying(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)
 
#Using all the functions
seed(1)
#Loading and preprocessing data
filename = 'leaf.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
#Converting class column to integers
look = str_column_to_int(dataset, len(dataset[0])-1)
#print(look)
#Normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
n_folds = 5
l_rate = 0.3
n_epoch = 100
n_hidden = 5
folds = cross_validation_split(dataset, n_folds)
for fold in folds:
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = list()
    for row in fold:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
        actual = [row[-1] for row in fold]
#print(actual)
print()
#for i in [0, 10, 100, 500, 1000, 10000]:       # For Training Dataset
#for i in [0.01, 0.2, 0.0001, 0.5, 1]:
result=applying(train_set,test_set, l_rate , n_epoch ,n_hidden)
#print(result)
acc=accuracy_metric(actual,result)
print("\nAccuracy : ",10*acc)

print(actual)
print(result)
mat = confusion_matrix(actual, result)
print(mat)
