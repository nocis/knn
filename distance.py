import numpy as np
def calDistance(point1, point2, type="Cosine"):
    left = point1[:len(point1)-1]
    right = point2[:len(point2)-1]
    if type == "Euclidean":
        return sum((left - right) ** 2) ** (1/2)
    elif type == "Manhattan":
        return sum(abs(left - right))
    elif type == "Minkowski":
        return sum((left - right) ** 4) ** (1/4)
    elif type == "Cosine":
        return 1.0 - np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
  