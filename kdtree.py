from distance import calDistance 
import numpy as np
class KdTree:
    """
    kdtree class, balanced binary tree, index can be computed
    """
    def __init__(self, data):
        self.data = self.buildTree(0, data)

    def buildTree(self, axis, data):
        n = len(data)
        if n <= 0:
            return None
        elif n <= 2:
            return data

        data = data[data[:,axis].argsort()]
        newAxis = (axis + 1) % len( data[0] - 1 )
        midPos = n // 2
        data[:midPos] = self.buildTree(newAxis, data[:midPos])
        data[midPos+1:] = self.buildTree(newAxis, data[midPos+1:])
        return data

    def printTree(self):
        print(self.data)

    def knn(self, k, point):
        self.knnResults = []
        self.knnDistances = []
        self.checkNearest(point, k, len(self.data) // 2, 0, len(self.data)-1, 0)
        self.knnResults = np.array(self.knnResults)
        self.knnDistances = np.array(self.knnDistances)

    def checkNearest(self, point, k, midPos, leftPos, rightPos, axis):
        newAxis = (axis + 1) % len( self.data[0] - 1 )
        near = 0
        far = 0
        leftNear = 0
        rightNear = 0
        leftFar = 0
        rightFar = 0
        if point[axis] - self.data[midPos][axis] > 0:
            near =  midPos + (rightPos - midPos) // 2 + 1
            far = leftPos + (midPos - leftPos) // 2
            leftNear = midPos+1
            rightNear = rightPos
            leftFar = leftPos
            rightFar = midPos-1
        else:
            far =  midPos + (rightPos - midPos) // 2 + 1
            near = leftPos + (midPos - leftPos) // 2
            leftNear = leftPos
            rightNear = midPos-1
            leftFar = midPos+1
            rightFar = rightPos

        if leftNear <= rightNear:
            self.checkNearest(point, k, near, leftNear, rightNear, newAxis)

        if len(self.knnResults) < k:
            self.knnResults.append(self.data[midPos])
            self.knnDistances.append(calDistance(self.data[midPos], point, "Cosine"))
            if leftFar <= rightFar:
                self.checkNearest(point, k, far, leftFar, rightFar, newAxis)
        else:
            maxidx = self.knnDistances.index(np.max(self.knnDistances))
            nearDistance = self.knnDistances[maxidx]
            midDistance = calDistance(self.data[midPos], point, "Cosine")
            if midDistance < nearDistance:
                self.knnResults[maxidx] = self.data[midPos]
                self.knnDistances[maxidx] = midDistance
                if leftFar <= rightFar:
                    self.checkNearest(point, k, far, leftFar, rightFar, newAxis)

            


