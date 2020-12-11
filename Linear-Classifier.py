import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    w = np.array([[1., 1., 1.]])
    """
    x = np.array([[2., 1., 1.],
                  [2., 2., 1.],
                  [1., 3., 1.],
                  [1., 0., 1.],
                  [1., 1., 1.],
                  [0., 2., 1.]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    """

    x = np.array([[1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1],
                  [1, 0, 1]], dtype='float32')
    y = np.array([1, 1, -1, -1])

    for i in range(100):
        print("iter=", i + 1)
        for j in range(4):
            pre = np.dot(w, x[j].T)[0]
            if pre * y[j] <= 0:
                w = w + y[j] * x[j]
            print(w)

    x_axis = np.linspace(0, 2.5, 100)
    y_axis = (w[0, 0]*x_axis + w[0, 2])/(-w[0, 1])
    
    plt.figure(figsize=(6, 5))
    plt.plot(x_axis, y_axis)
    # plt.subplot(122)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title("Perceptron")
    plt.savefig('./cannot.png')
    plt.show()

