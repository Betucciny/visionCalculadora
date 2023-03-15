import numpy as np
import cv2
import matplotlib.pyplot as plt
_lambda = 255


def FP_iter(M1, func, args=None):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros((high, width, 3), dtype=np.uint8)
    for k in range(3):
        for i in range(high):
            for j in range(width):
                M2[i, j, k] = func(M1[i, j, k])
    return M2


def FP_Iden(M1, args=None):
    cv2.imread('output.png', FP_iter(M1, lambda x: x))


def FP_Neg(M1, args=None):
    cv2.imread('output.png', FP_iter(M1, lambda x: _lambda - x))


def FP_Grises(M1, args=None):
    M2 = np.zeros(M1.shape, dtype=np.uint8)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            M2[i, j] = np.mean(M1[i, j])
    cv2.imwrite('output.png', M2)


def FP_Bin(M1, args=_lambda//2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda if x > umbral else 0))



def FP_BinInv(M1, args=_lambda//2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: 0 if x > umbral else _lambda))


def FP_Log(M1, args=1):
    a = args
    c = _lambda / np.log(1 + a*_lambda)
    cv2.imwrite('output.png', FP_iter(M1, lambda x: c * np.log(1 + a*x)))


def FP_gamma(M1, args=0.5):
    gamma = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (x / _lambda)**gamma))


def FP_Seno(M1, args=None):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * np.sin(x*np.pi / (2*_lambda))))


def FP_Coseno(M1, args=None):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (1 - np.cos(x*np.pi / (2*_lambda)))))


def FR_Border(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros((high, width, 3), dtype=np.uint8)
    for k in range(3):
        for i in range(high):
            for j in range(width-1):
                M2[i, j, k] = M1[i, j+1, k] - M1[i, j, k]
    cv2.imwrite('output.png', M2)

