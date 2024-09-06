#!/usr/bin/env python3

import numpy as np
import sys
import random
from matplotlib import image
from matplotlib import pyplot as plt
from scipy.special import expit

def plotall(w):

    fig, ax = plt.subplots(6,12,figsize=(20,10))
    for i in range(6):
        for j in range(12):
            image = w[6*i+j,:]
            hi = np.max(image)
            lo = np.min(image)
            image = (255.0*(image - lo) / (hi - lo)).astype(np.uint8)
            ax[i,j].imshow(image.reshape(20,20),cmap='gray')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":

    a1 = image.imread("digits.png")
    a2 = a1 .reshape(50, 20, 100, 20)
    a3 = a2.transpose([0,2,1,3])
    a4 = a3.reshape(10,5,100,20,20)
    a5 = a4.reshape(10,500,400)
    digits = a5[ : , np.random.permutation(500) , :]

    train = digits[0:10,0:400,0:400]
    test  = digits[0:10,400:500,0:400]

    alpha = 0.25
    hidden = 100

    w1 = np.random.randn(400*hidden).reshape(hidden,400)
    w2 = np.random.randn(10*hidden).reshape(10,hidden)
    iter   = 0

    while iter < 100000:
        iter += 1

        k = random.randrange(10)
        j = random.randrange(400)
        x = train[k,j]
        y = np.zeros(10)
        y[k] = 1.0

        z1 = np.matmul(w1,x)    # a2 = expit(w2@expit(w1@x))
        a1 = expit(z1)
        z2 = np.matmul(w2,a1)   
        a2 = expit(z2)          # a2 is yhat

        delta3 = (a2 - y)
        delta2 = np.matmul(w2.T, delta3)

        dw2 = np.outer(delta3, a1)
        w2 -= dw2 * alpha

        dw1 = np.outer(delta2, x)
        w1 -= dw1 * alpha

        alpha *= 0.9999

    plotall(w1)    # to view hidden layer nodes as image

    good = 0
    total = 0
    tally = np.zeros(100).astype(np.int32).reshape(10,10)
    for i in range(10):
        for j in range(100):
            x  = test[i,j]
            z1 = np.matmul(w1,x)
            a1 = expit(z1)
            z2 = np.matmul(w2,a1)
            a2 = expit(z2)

            k  = np.argmax(a2)
            if k == i:
                good += 1
            total += 1
            tally[i,k] += 1
    rate = good / total

    print(alpha, rate, good, total,end='\n\n')
    for i in range(10):
        for j in range(10):
            print(f'{tally[i,j]:6d}',end='')
        print()

