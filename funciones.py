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


def write_histogram(hist, filename='histogram.png'):
    plt.plot(hist)
    plt.savefig(filename)
    plt.clf()


def write_image(M1, filename='output.png'):
    cv2.imwrite(filename, M1)


def normalize(M1):
    return np.floor(_lambda * (M1 - np.min(M1)) / (np.max(M1) - np.min(M1)))

# 2. Filtros Puntuales
def FP_Iden(M1):
    return FP_iter(M1, lambda x: x)

# Filtro negativo
def FP_Neg(M1):
    return FP_iter(M1, lambda x: _lambda - x)

# Filtro Gris (R+G+B)/3
def FP_Grises(M1):
    M2 = np.zeros(M1.shape, dtype=np.uint8)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            M2[i, j] = np.mean(M1[i, j])
    return M2

# filtro binario
def FP_Bin(M1, args=_lambda // 2):
    umbral = args
    return FP_iter(M1, lambda x: _lambda if x > umbral else 0)

# filtro binario inverso
def FP_BinInv(M1, args=_lambda // 2):
    umbral = args
    return FP_iter(M1, lambda x: 0 if x > umbral else _lambda)

# Filtro aclarado logaritmico, rango dinamico
# Filtro rango dinamico parametrizado (alfa)
def FP_Log(M1, args=1):
    a = args
    c = _lambda / np.log(1 + a * _lambda)
    return FP_iter(M1, lambda x: c * np.log(1 + a * x))


# Filtro correccion Gamma
def FP_gamma(M1, args=0.5):
    gamma = args
    return FP_iter(M1, lambda x: _lambda * (x / _lambda) ** gamma)

# Filtro funcion seno
def FP_Seno(M1):
    return FP_iter(M1, lambda x: _lambda * np.sin(x * np.pi / (2 * _lambda)))

# Filtro funcion coseno
def FP_Coseno(M1):
    return FP_iter(M1, lambda x: _lambda * (1 - np.cos(x * np.pi / (2 * _lambda))))

# Filtro aclarado exponencial
def FP_Exp_Acl(M1, args=1):
    a = args
    A = _lambda / (1 - np.exp(-a))
    return FP_iter(M1, lambda x: A * (1 - np.exp(-a * x / _lambda)))

# Filtro oscurecimiento exponencial
def FP_Exp_Osc(M1, args=1):
    a = args
    A = _lambda / (np.exp(a)-1)
    return FP_iter(M1, lambda x: A * (np.exp(a * x / _lambda) - 1))


# Filtro sigmoide seno (Revisar)
def FP_Sigmoid_Sin(M1, args=(10, 0.5)):
    a, b = args
    return FP_iter(M1, lambda x: _lambda * (1 / (1 + np.exp(-a * (np.sin(b * (x / 255) - np.pi / 2))))))

# Filtro sigmoide tangente hiperbólica. (Revisar)
def FP_Sigmoid_Tanh(M1, args=(5, 0.5)):
    a, b = args
    return FP_iter(M1, lambda x: _lambda * (1 / (1 + np.exp(-a * np.tanh(b * (x / 255 - 0.5))))))

# 3. Ecualizacion por histograma (Revisar)
def histogram_equalization(M1):
    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(M1, cv2.COLOR_BGR2GRAY)
    # Calcular histograma de la imagen original
    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # Ecualizar la imagen
    img_equalized = cv2.equalizeHist(gray)
    # Calcular histograma de la imagen ecualizada
    hist_equalized = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
    return [cv2.cvtColor(M1, cv2.COLOR_BGR2RGB), hist_original, cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB), hist_equalized]


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

    return [cv2.cvtColor(M1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2RGB)]


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


# (a) Detector de orillas Sobel de 3x3 (Revisar)
def detect_edges_sobel(img):
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar Sobel en X e Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud de los gradientes y la dirección
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x)

    # Normalizar las imágenes para el despliegue
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    direction_norm = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return [{sobel_x, cmap='gray'}, {sobel_y, cmap='gray'}, {magnitude_norm, cmap='gray'}]

    # Mostrar las imágenes de bordes en X, Y y la magnitud
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(sobel_x, cmap='gray')
    axs[0].set_title('Bordes en X')
    axs[1].imshow(sobel_y, cmap='gray')
    axs[1].set_title('Bordes en Y')
    axs[2].imshow(magnitude_norm, cmap='gray')
    axs[2].set_title('Magnitud de los bordes')
    plt.show()


# (b) Detector Canny (Revisar)
def detect_edges_canny(img):
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar Sobel en X e Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud de los gradientes y la dirección
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x)

    # Normalizar las imágenes para el despliegue
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    direction_norm = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Aplicar Canny
    canny = cv2.Canny(gray, 100, 200)

    # Mostrar las imágenes de bordes en X, Y y la magnitud
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(magnitude_norm, cmap='gray')
    axs[0].set_title('Magnitud de los bordes')
    axs[1].imshow(direction_norm, cmap='gray')
    axs[1].set_title('Dirección de los bordes')
    axs[2].imshow(canny, cmap='gray')
    axs[2].set_title('Canny')
    plt.show()
