
from kdtree import KdTree
from knn import knn_naive
from sklearn.model_selection import KFold
import random
import numpy as np
import math

def accuracy(ytrue, ypred):
	correct = 0
	for i in range(len(ytrue)):
		if ytrue[i] == ypred[i]:
			correct += 1
	return float(correct) / float(len(ytrue)) * 100.0

def naiveWeight(distances):
    results = np.zeros(distances.shape)
    for i in range(len(distances)):
        results[i] = 1.0 / (distances[i] + 0.1)
    return results

def knnClassifier(train, test, k, kdtree=False, weight=naiveWeight):
    predictions = test
    probs = []
    kdTreeKnn = None
    kNeighbors = None
    kWeights = None
    if kdtree:
        kdTreeKnn = KdTree(train)

    for i in range(len(test)):
        for j in range(len(test[0])):
            predictions[i][j] = test[i][j]
        if kdtree:
            kdTreeKnn.knn(k, predictions[i])
            kNeighbors = kdTreeKnn.knnResults
            kWeights = kdTreeKnn.knnDistances
        else:
            kNeighbors, kWeights = knn_naive(k, train, predictions[i])

        kWeights = weight(kWeights)
        sumWeights = 0
        sumPred = 0
        for j in range(k):
            sumWeights += kWeights[j]
            sumPred += kWeights[j] * kNeighbors[j][-1]

        predictions[i][-1] = round(sumPred / sumWeights)
        probs.append(list(predictions[i]))
        probs[i][-1] = sumPred / sumWeights
    return (predictions, probs)

def evaluateKnn(data, n, *args):
    kf = KFold(n_splits=n, shuffle=True)
    tmp = np.array(data)
    scores = []
    itr = 0
    for train_index, test_index in kf.split(data):
        train, true = tmp[train_index].tolist(), tmp[test_index].tolist()
        test = []
        for row in true:
            rowCopy = list(row)
            rowCopy[-1] = 0
            test.append(rowCopy)

        pred ,_ = knnClassifier(np.array(train), np.array(test), *args)
        yTrue = [row[-1] for row in true]
        yPred = [row[-1] for row in pred]
        acc = accuracy(yTrue, yPred)
        scores.append(acc)
        
        itr += 1
        #print(itr)   
    return scores
