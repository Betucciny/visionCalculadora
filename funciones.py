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

# 2. Filtros Puntuales
def FP_Iden(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: x))

# Filtro negativo
def FP_Neg(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda - x))

# Filtro Gris (R+G+B)/3
def FP_Grises(M1):
    M2 = np.zeros(M1.shape, dtype=np.uint8)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            M2[i, j] = np.mean(M1[i, j])
    cv2.imwrite('output.png', M2)

# filtro binario
def FP_Bin(M1, args=_lambda // 2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda if x > umbral else 0))

# filtro binario inverso
def FP_BinInv(M1, args=_lambda // 2):
    umbral = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: 0 if x > umbral else _lambda))

# Filtro aclarado logaritmico, rango dinamico
def FP_Log(M1, args=1):
    a = args
    c = _lambda / np.log(1 + a * _lambda)
    cv2.imwrite('output.png', FP_iter(M1, lambda x: c * np.log(1 + a * x)))


# Filtro rango dinamico parametrizado (alfa)



# Filtro correccion Gamma
def FP_gamma(M1, args=0.5):
    gamma = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (x / _lambda) ** gamma))

# Filtro funcion seno
def FP_Seno(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * np.sin(x * np.pi / (2 * _lambda))))

# Filtro funcion coseno
def FP_Coseno(M1):
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (1 - np.cos(x * np.pi / (2 * _lambda)))))

# Filtro aclarado exponencial
def FP_Exp_Acl(M1, args=1):
    a = args
    A = _lambda / (1 - np.exp(-a))
    cv2.imwrite('output.png', FP_iter(M1, lambda x: A * (1 - np.exp(-a * x / _lambda))))

# Filtro oscurecimiento exponencial
def FP_Exp_Osc(M1, args=1):
    a = args
    A = _lambda / (np.exp(a)-1)
    cv2.imwrite('output.png', FP_iter(M1, lambda x: A * (np.exp(a * x / _lambda) - 1)))


# Filtro sigmoide seno (Revisar)
def FP_Sigmoid_Sin(M1, args=(10, 0.5)):
    a, b = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (1 / (1 + np.exp(-a * (np.sin(b * (x / 255) - np.pi / 2)))))))



# Filtro sigmoide tangente hiperbólica. (Revisar)
def FP_Sigmoid_Tanh(M1, args=(5, 0.5)):
    a, b = args
    cv2.imwrite('output.png', FP_iter(M1, lambda x: _lambda * (1 / (1 + np.exp(-a * np.tanh(b * (x / 255 - 0.5)))))))




# 3. Ecualizacion por histograma (Revisar)
def histogram_equalization(M1):
    M1 = cv2.cvtColor(M1, cv2.COLOR_BGR2GRAY)
    # Calcular el histograma de la imagen de entrada
    hist, bins = np.histogram(M1.flatten(), 256, [0, 256])

    # Calcular la función de distribución acumulativa (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Ecualizar la imagen
    img_equalized = np.interp(M1.flatten(), bins[:-1], cdf_normalized).reshape(M1.shape)

    # Calcular el histograma de la imagen ecualizada
    hist_eq, bins_eq = np.histogram(img_equalized.flatten(), 256, [0, 256])


# 4. Filtrado espacial (Revisar)
def spatial_filter(M1, mask):
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(M1, cv2.COLOR_BGR2GRAY)

    # Obtener las dimensiones de la imagen y la máscara
    rows, cols = img_gray.shape[:2]
    mask_rows, mask_cols = mask.shape[:2]

    # Calcular el tamaño del borde para la máscara
    border = mask_rows // 2

    # Crear una matriz de ceros para la imagen filtrada
    img_filtered = np.zeros_like(img_gray)

    # Recorrer la imagen y aplicar la máscara
    for i in range(border, rows - border):
        for j in range(border, cols - border):
            roi = img_gray[i - border : i + border + 1, j - border : j + border + 1]
            img_filtered[i, j] = (roi * mask).sum()

    # Normalizar la imagen filtrada
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)

# Definir una mascara: Esta mascara aplica un filtro de realce de bordes a la imagen.
mask = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


# 5. Detección de orillas (Por completar)
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

