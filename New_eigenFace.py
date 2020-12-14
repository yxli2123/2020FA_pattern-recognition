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
    :param data_number:the file number in the data folder
    :param size: image size with (Height, Width)
    :return: data matrix with shape (Height*Width, data_number), label matrix with shape (data_number, )
    """
    # get the images in order
    dataList = [f for f in sorted(listdir(dataset_path))]
    # resize images into same shape
    H, W = size
    dataMatrix = np.empty((H * W, data_number), dtype='float32')
    label = np.empty(data_number, dtype='uint8')
    # load images randomly
    randomList = random.sample(range(0, data_number), data_number)
    # load images in order
    # randomList = np.arange(data_number)
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
    # load images X is the raw data , and y is label
    X_train, y_train = loadData('./Dataset/train/', trainDataNumber)
    X_test, y_test = loadData('./Dataset/test/', testDataNumber)

    # compute the meanFace
    X_mean = X_train.mean(axis=1)
    X_mean = X_mean.reshape(X_mean.shape[0], 1)
    meanImage = X_mean.reshape((190, 230), order='F').T
    meanImage = Image.fromarray(np.uint8(meanImage))
    meanImage.save('./meanFace.png')

    # compute the covariance of X and X.T
    X = X_train - X_mean
    Cov = np.dot(X, X.T)
    
    # compute its eigenvector and eigenvalue
    # using a trick to reduce the complexity
    Cov_ = np.dot(X.T, X)
    eigenValue, eigenVector_ = np.linalg.eig(Cov_)
    eigenVector = np.dot(X, eigenVector_)
    
    # we try different k low-dimension representation of raw image data
    for k in range(3, 13):
        
        # find the max k eigenvalue and their according eigenfaces
        k_eigenValue_index = (np.argsort(np.absolute(eigenValue))[::-1])[0:k]
        eigenFaces = eigenVector[:, k_eigenValue_index]
        
        # decompose X by eigendaces and calculate the weights
        weight = np.dot(eigenFaces.T, X)  # (i, j) the wight of ith eigenface in jth image
        
        # normalize the weight
        weight = (weight - weight.min(axis=0)) / (weight.max(axis=0) - weight.min(axis=0))
        
        # visualize eigenfaces
        for i in range(k):
            max = eigenFaces[:, i].max()
            min = eigenFaces[:, i].min()
            eigenFaces[:, i] = 255 * (eigenFaces[:, i] - min) / (max - min)
            faceImage = Image.fromarray(np.uint8(eigenFaces[:, i].reshape((190, 230), order='F').T))
            faceImage.save('./Dataset/eigenFace/' + str(i) + '.png')
        
        # decompose test images by eigendaces and calculate the weights
        weight_test = np.dot(eigenFaces.T, X_test - X_mean)
        
        # normalize the weight
        weight_test = (weight_test - weight_test.min(axis=0)) / (weight_test.max(axis=0) - weight_test.min(axis=0))
        
        num_correct = 0
        
        # compare low-dimension representation of X_test and X_train
        for i in range(X_test.shape[1]):
            
            # compute  Euclidean distance
            distance = (np.square(weight - weight_test[:, i].reshape(weight.shape[0], 1))).sum(axis=0)
            
            # sort the distance ascendingly
            index = np.uint8(np.argsort(distance))
            y_pred = y_train[index]
            """
            We could add a threshold here to determine if a image is in the train set
            or it is a human face image.
            """
            # y_pred[0] is the minimam
            if y_pred[0] == y_test[i]:
                num_correct += 1
        print(num_correct)


