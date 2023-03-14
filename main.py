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
    return FP_iter(M1, lambda x: x)


def FP_Neg(M1, args=None):
    return FP_iter(M1, lambda x: _lambda - x)


def FP_Bin(M1, args=_lambda//2):
    umbral = args
    return FP_iter(M1, lambda x: _lambda if x > umbral else 0)


def FP_BinInv(M1, args=_lambda//2):
    umbral = args
    return FP_iter(M1, lambda x: 0 if x > umbral else _lambda)


def FP_Log(M1, args=1):
    a = args
    c = _lambda / np.log(1 + a*_lambda)
    return FP_iter(M1, lambda x: c * np.log(1 + a*x))


def FP_gamma(M1, args=0.5):
    gamma = args
    return FP_iter(M1, lambda x: _lambda * (x / _lambda)**gamma)


def FP_Seno(M1, args=None):
    return FP_iter(M1, lambda x: _lambda * np.sin(x*np.pi / (2*_lambda)))


def FP_Coseno(M1, args=None):
    return FP_iter(M1, lambda x: _lambda * (1 - np.cos(x*np.pi / (2*_lambda))))


def read(ruta):
    list_func = [(FP_Iden, 'Identidad', []), (FP_Neg, 'Negativo', []), (FP_gamma, 'Gamma', [0.5]),
                 (FP_Log, 'Log', [1]), (FP_Seno, 'Sin', []), (FP_Coseno, 'Cos', [])]
    img = cv2.imread(ruta)
    for i, cont in enumerate(list_func):
        func, title, args = cont
        imgE = func(img, args[0])
        cv2.imwrite('{}.png'.format(title), imgE)


def main():
    read('rumbling.png')


if __name__ == '__main__':
    main()

