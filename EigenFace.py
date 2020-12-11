import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from os import listdir
from os.path import join


def loadData(dataset_path: str, sample_number: int, class_number: int, withLight=True):
    """

    :param withLight: True if the image is lit
    :param dataset_path: the path of the folder containing data(training data or test data)
    :param sample_number: the sample face number of a person, e.g. with glass, smile, sad ...
    :param class_number: number of the  people, >1
    :return: C x S x N Numpy matrix, C: class number, S: sample number, N: length of flatten raw face image
             C x 1 x N Numpy matrix, C: class number, N: length of flatten raw face image
    """

    dataList = [f for f in sorted(listdir(dataset_path))]
    H = 190
    W = 230
    dataMatrix = np.zeros((class_number, sample_number, H * W), dtype='float32')
    meanMatrix = np.zeros((class_number, 1, H * W), dtype='float32')
    for c_ in range(class_number):
        for s in range(sample_number):
            image = Image.open(join(dataset_path, dataList[c_ * sample_number + s]))
            image = image.resize((H, W))
            image_flat = np.asarray(image, dtype='float32')
            image_flat = image_flat.flatten()
            dataMatrix[c_, s, :] = image_flat
        if not withLight:
            dataMatrix[c_, 0:sample_number - 3, :] = np.delete(dataMatrix[c_], [0, 1, 4], axis=0)
            meanMatrix[c_, 0] = dataMatrix[c_, 0:sample_number - 3, :].mean(axis=0)
        else:
            meanMatrix[c_, 0] = dataMatrix[c_].mean(axis=0)
    if withLight:
        return dataMatrix, meanMatrix
    else:
        return dataMatrix[:, 0:sample_number-3, :], meanMatrix


if __name__ == '__main__':
    trainClassNumber = 15
    testSampleNumber = 10
    # load the data into CxSxF and the mean
    trainMatrix, meanTrainMatrix = loadData('./Dataset/train/', 9, trainClassNumber, withLight=False)
    testMatrix, meanTestMatrix = loadData('./Dataset/test/', 1, testSampleNumber)

    # using PCA to find weight of each class; TRAIN LOOP
    dimension = 3
    projectionMatrixList = []
    trainSetWeightList = []
    for c in range(trainClassNumber):
        pca = PCA(n_components=dimension, whiten=True)
        pca.fit(trainMatrix[c] - meanTrainMatrix[c])
        projectionMatrixList.append(pca.components_.T)  # N x dimension
        temp = trainMatrix[c]
        trainSetWeightList.append(np.dot(temp - meanTrainMatrix[c], pca.components_.T))  # S x dimension

    # TEST LOOP
    compareMatrix = np.zeros(testSampleNumber*trainClassNumber).reshape(testSampleNumber, trainClassNumber)
    k = 1
    for t in range(testSampleNumber):
        for c in range(trainClassNumber):
            weight = np.dot(testMatrix[t] - meanTrainMatrix[c], projectionMatrixList[c])  # 1 x dimension
            distanceMatrix = np.square(trainSetWeightList[c] - weight)
            distanceMatrix = np.sum(distanceMatrix, axis=1)
            k_nearest_distance = (np.sort(distanceMatrix))[0:k]
            averageDistance = np.sqrt(k_nearest_distance.mean())
            compareMatrix[t, c] = averageDistance

    bestCompare = np.argmin(compareMatrix, axis=1) + 1
    print(bestCompare)
