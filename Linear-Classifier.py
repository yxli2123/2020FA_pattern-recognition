import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # initialize the weight augmented matrix
    w = np.array([[1., 1., 1.]])
    
    # feed training data and their labels
    x = np.array([[2., 1., 1.],
                  [2., 2., 1.],
                  [1., 3., 1.],
                  [1., 0., 1.],
                  [1., 1., 1.],
                  [0., 2., 1.]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    
    # train classifier for 20 times
    for i in range(20):
        print("iter=", i + 1)
        for j in range(y.shape[0]):
            y_pre = np.dot(w, x[j].T)[0]
            if y_pre * y[j] <= 0:
                w = w + y[j] * x[j]
            print(w)
            
    # ready to plot th line of linear classifier
    x_axis = np.linspace(0, 2.5, 100)
    y_axis = (w[0, 0]*x_axis + w[0, 2])/(-w[0, 1])
    
    # initialize a figure
    plt.figure(figsize=(6, 5))
    plt.plot(x_axis, y_axis)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title("Linear Classifier")
    
    plt.savefig('./result.png')
    plt.show()

