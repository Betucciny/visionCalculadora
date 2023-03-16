import numpy as np
# Máscaras en el dominio espacial
mask1 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

mask2 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])

mask3 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) * (1/9)

mask4 = np.array([
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, 0]
])

mask5 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

mask6 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

mask7 = np.array([
    [0, 0, 0],
    [-1, 1, 0],
    [0, 0, 0]
])

mask8 = np.array([
    [0, -1, 0],
    [0, 2, 0],
    [0, -1, 0]
])

# Lista de máscaras
mask_list = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]
# Descripciones de las máscaras
mask_desc = [
    "Realce de bordes",
    "Realce de detalles",
    "Filtro de promedio",
    "Detector de bordes verticales",
    "Detector de bordes horizontales",
    "Detector de bordes en diagonal (45 grados)",
    "Detector de bordes en diagonal (135 grados)",
    "Detector de bordes de línea"
]

