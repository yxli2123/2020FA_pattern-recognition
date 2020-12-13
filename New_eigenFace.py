import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from os import listdir
from os.path import join
import random
import re


def loadData(dataset_path: str, data_number: int, size=(190, 230)):
    """
    :param dataset_path: the path of the folder containing data(training data or test data)
    :param data_number:
    :param size:
    :return:
    """
    # get the images in order
    dataList = [f for f in sorted(listdir(dataset_path))]
    # resize images into same shape
    H, W = size
    dataMatrix = np.empty((H * W, data_number), dtype='float32')
    label = np.empty(data_number, dtype='uint8')
    # load images randomly
    # randomList = random.sample(range(0, data_number), data_number)
    randomList = np.arange(data_number)
    for i in range(data_number):
        image = Image.open(join(dataset_path, dataList[randomList[i]]))
        image = image.resize((H, W))
        image_flat = np.asarray(image, dtype='float32')
        image_flat = image_flat.flatten()
        dataMatrix[:, i] = image_flat

        label_str = re.findall(r"\d+", dataList[randomList[i]])
        label[i] = int(label_str[0])

    return dataMatrix, label


if __name__ == '__main__':
    trainDataNumber = 7*15
    testDataNumber = 10

    X_train, y_train = loadData('./Dataset/train/', trainDataNumber)
    X_test, y_test = loadData('./Dataset/test/', testDataNumber)


    X_mean = X_train.mean(axis=1)
    meanImage = X_mean.reshape((190, 230), order='F').T
    meanImage = Image.fromarray(np.uint8(meanImage))
    meanImage.save('./mean.png')

    X_mean = X_mean.reshape(X_mean.shape[0], 1)
    X = X_train - X_mean
    Cov = np.dot(X, X.T)
    Cov_ = np.dot(X.T, X)
    eigenValue, eigenVector_ = np.linalg.eig(Cov_)
    eigenVector = np.dot(X, eigenVector_)

    for d in range(10, 11):
        k_eigenValue_index = (np.argsort(np.absolute(eigenValue))[::-1])[0:d]
        eigenFaces = eigenVector[:, k_eigenValue_index]

        weight = np.dot(eigenFaces.T, X)  # (i, j) the wight of ith eigenface in jth image
        weight = (weight - weight.min(axis=0)) / (weight.max(axis=0) - weight.min(axis=0))

        for i in range(d):
            max = eigenFaces[:, i].max()
            min = eigenFaces[:, i].min()
            eigenFaces[:, i] = 255 * (eigenFaces[:, i] - min) / (max - min)
            faceImage = Image.fromarray(np.uint8(eigenFaces[:, i].reshape((190, 230), order='F').T))
            faceImage.save('./Dataset/eigenFace/' + str(i) + '.png')

        weight_test = np.dot(eigenFaces.T, X_test - X_mean)
        weight_test = (weight_test - weight_test.min(axis=0)) / (weight_test.max(axis=0) - weight_test.min(axis=0))
        num = 0
        for i in range(X_test.shape[1]):
            distance = (np.square(weight - weight_test[:, i].reshape(weight.shape[0], 1))).sum(axis=0)
            index = np.uint8(np.argsort(distance))
            result = y_train[index]
            if result[0] == y_test[i]:
                num += 1
        print(num)


