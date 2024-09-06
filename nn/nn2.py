#!/usr/bin/env python3

import sys
from random import randrange
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.special import expit
import PIL as pil


def main():
    a1 = image.imread("digits.png")
    a2 = a1.reshape(
        50, 20, 100, 20
    )  # 50 rows of digits, 20 pixels high, 100 columns of digts, 20 pixels wide
    a3 = a2.transpose([0, 2, 1, 3])
    a4 = a3.reshape(10, 5, 100, 20, 20)  # 10 digits, 5 rows, 100 columns, 20 by 20
    a5 = a4.reshape(
        10, 500, 400
    )  # 10 digits, 500 pictures of each, 400 pixels per picture

    images = a5[:, np.random.permutation(500), :]

    train = images[0:10, 0:400, 0:400]
    test = images[0:10, 400:500, 0:400]

    alpha = 0.5
    w = np.random.randn(4000).reshape(10, 400)  # rows = # outputs, columns = # inputs

    right = wrong = 0

    for i in range(250000):
        digit = randrange(10)
        pictno = randrange(400)

        x = train[digit, pictno]

        y = np.zeros(10)
        y[digit] = 1

        tmp = w @ x  # matrix multiplication
        yhat = expit(
            tmp
        )  # apply the activation function: sigmoid, or logistic functions

        predict = np.argmax(yhat)

        if predict == digit:
            right += 1
        else:
            wrong += 1

        delta = yhat - y  # error values for each of the 10 output nodes
        dw = np.outer(delta, x)
        w -= alpha * dw

        if i % 1000 == 0:
            alpha *= 0.98

    print(right, wrong)

    good = 0
    total = 0

    tally = np.zeros(100).astype(np.int32).reshape(10, 10)
    for i in range(10):
        for j in range(100):
            x = test[i, j]
            z = np.matmul(w, x)
            yhat = expit(z)

            predict = np.argmax(yhat)
            if predict == i:
                good += 1
            total += 1
            tally[i, predict] += 1
    rate = good / total

    for i in range(10):
        for j in range(10):
            print(f"{tally[i,j]:6d}", end="")
        print()
    print(f"final alpha: {alpha:9.6f},  fraction correct: {rate:9.6f}")
    np.save("wgts.npy", w)


if __name__ == "__main__":
    main()
