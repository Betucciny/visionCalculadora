import matplotlib.pyplot as plt
import numpy as np
import cv2

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


def normalize(M1):
    return np.floor(_lambda * (M1 - np.min(M1)) / (np.max(M1) - np.min(M1)))


def FP_Iden(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: x))


def FP_Neg(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda - x))


def FP_Grises(M1):
    M2 = np.zeros(M1.shape, dtype=np.uint8)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            M2[i, j] = np.mean(M1[i, j])
    cv2.imwrite('output.png', M2)


def FP_Bin(M1, args=_lambda // 2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda if x > umbral else 0))


def FP_BinInv(M1, args=_lambda // 2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: 0 if x > umbral else _lambda))


def FP_Log(M1, args=1):
    a = args
    c = _lambda / np.log(1 + a * _lambda)
    cv2.imwrite('output.png', FP_iter(M1, lambda x: c * np.log(1 + a * x)))


def FP_gamma(M1, args=0.5):
    gamma = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (x / _lambda) ** gamma))


def FP_Seno(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * np.sin(x * np.pi / (2 * _lambda))))


def FP_Coseno(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (1 - np.cos(x * np.pi / (2 * _lambda)))))


def FP_Exp_Acl(M1, args=1):
    a = args
    A = _lambda / (1 - np.exp(-a))
    cv2.imwrite('output.png', FP_iter(M1, lambda x: A * (1 - np.exp(-a * x / _lambda))))


def FP_Exp_Osc(M1, args=1):
    a = args
    A = _lambda / (np.exp(a)-1)
    cv2.imwrite('output.png', FP_iter(M1, lambda x: A * (np.exp(a * x / _lambda) - 1)))





def FR_Border_X(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros((high, width, 3), dtype=np.uint8)
    for k in range(3):
        for i in range(high):
            for j in range(width - 1):
                M2[i, j, k] = M1[i, j + 1, k] - M1[i, j, k]
    M2 = normalize(M2)
    cv2.imwrite('output.png', M2)


def FR_Border_Y(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros((high, width, 3), dtype=np.uint8)
    for i in range(high - 1):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = M1[i][j + 1][k] - M1[i, j, k]
    M2 = normalize(M2)
    cv2.imwrite('output.png', M2)


def Conv(M1, kernel):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros((high, width, 3), dtype=np.uint8)
    for i in range(high-3):
        for j in range(width-3):
            for k in range(3):
                M2[i, j, k] = np.sum(M1[i:i + 3, j:j + 3, k] * kernel)
    # M2 = normalize(M2)
    cv2.imwrite('output.png', M2)

