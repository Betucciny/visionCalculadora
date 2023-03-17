from tkinter import *
from tkinter.messagebox import showinfo
from funciones import *
from tkinter import filedialog as fd
from mask_mem import *


def abrir_imagen():
    global imagen
    ruta = fd.askopenfilename(title = "Abrir Imagen", filetypes = (("png files","*.png"),("all files","*.*")))
    imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)
    showinfo(title="Confirmacion", message="Imagen Guardada")


def show_confirmacion1():
    showinfo(title="Confirmacion", message="Imagen Editada con exito")


imagen = cv2.imread('rumbling.png')

root = Tk()
root.title("Calculadora de Imagenes")
root.geometry("400x8")

menu_principal = Menu(root)


archivo = Menu(menu_principal, tearoff=0)
archivo.add_command(label="Abrir Imagen", command=abrir_imagen)

filtro_puntual = Menu(menu_principal, tearoff=0)

filtro_puntual.add_command(label="Identidad", command=lambda : (write_image(FP_Iden(imagen)), show_confirmacion1()))
filtro_puntual.add_command(label="Negativo", command=lambda : (write_image(FP_Neg(imagen)), show_confirmacion1()))
filtro_puntual.add_command(label="Grises", command=lambda : (write_image(FP_Grises(imagen)), show_confirmacion1()))


binario = Menu(filtro_puntual, tearoff=0)
binario.add_command(label=f'Umbral: {0}', command=lambda: (write_image(FP_Bin(imagen, args=0)), show_confirmacion1()))
binario.add_command(label=f'Umbral: {51}', command=lambda: (write_image(FP_Bin(imagen, args=51)), show_confirmacion1()))
binario.add_command(label=f'Umbral: {102}', command=lambda: (write_image(FP_Bin(imagen, args=102)), show_confirmacion1()))
binario.add_command(label=f'Umbral: {153}', command=lambda: (write_image(FP_Bin(imagen, args=153)), show_confirmacion1()))
binario.add_command(label=f'Umbral: {204}', command=lambda: (write_image(FP_Bin(imagen, args=204)), show_confirmacion1()))

binario_inverso = Menu(filtro_puntual, tearoff=0)
binario_inverso.add_command(label=f'Umbral: {0}', command=lambda: (write_image(FP_BinInv(imagen, args=0)), show_confirmacion1()))
binario_inverso.add_command(label=f'Umbral: {51}', command=lambda: (write_image(FP_BinInv(imagen, args=51)), show_confirmacion1()))
binario_inverso.add_command(label=f'Umbral: {102}', command=lambda: (write_image(FP_BinInv(imagen, args=102)), show_confirmacion1()))
binario_inverso.add_command(label=f'Umbral: {153}', command=lambda: (write_image(FP_BinInv(imagen, args=153)), show_confirmacion1()))
binario_inverso.add_command(label=f'Umbral: {204}', command=lambda: (write_image(FP_BinInv(imagen, args=204)), show_confirmacion1()))

logaritmico = Menu(filtro_puntual, tearoff=0)
logaritmico.add_command(label=f'Alfa: {1}', command=lambda: (write_image(FP_Log(imagen, args=1)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {2}', command=lambda: (write_image(FP_Log(imagen, args=2)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {3}', command=lambda: (write_image(FP_Log(imagen, args=3)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {4}', command=lambda: (write_image(FP_Log(imagen, args=4)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {5}', command=lambda: (write_image(FP_Log(imagen, args=5)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {6}', command=lambda: (write_image(FP_Log(imagen, args=6)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {7}', command=lambda: (write_image(FP_Log(imagen, args=7)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {8}', command=lambda: (write_image(FP_Log(imagen, args=8)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {9}', command=lambda: (write_image(FP_Log(imagen, args=9)), show_confirmacion1()))
logaritmico.add_command(label=f'Alfa: {10}', command=lambda: (write_image(FP_Log(imagen, args=10)), show_confirmacion1()))

gamma = Menu(filtro_puntual, tearoff=0)
gamma.add_command(label=f'Gamma: {0.25}', command=lambda: (write_image(FP_gamma(imagen, args=0.25)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {0.5}', command=lambda: (write_image(FP_gamma(imagen, args=0.5)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {0.75}', command=lambda: (write_image(FP_gamma(imagen, args=0.75)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {1}', command=lambda: (write_image(FP_gamma(imagen, args=1)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {1.25}', command=lambda: (write_image(FP_gamma(imagen, args=1.25)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {1.5}', command=lambda: (write_image(FP_gamma(imagen, args=1.5)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {1.75}', command=lambda: (write_image(FP_gamma(imagen, args=1.75)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {2}', command=lambda: (write_image(FP_gamma(imagen, args=2)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {2.25}', command=lambda: (write_image(FP_gamma(imagen, args=2.25)), show_confirmacion1()))
gamma.add_command(label=f'Gamma: {2.5}', command=lambda: (write_image(FP_gamma(imagen, args=2.5)), show_confirmacion1()))

expa = Menu(filtro_puntual, tearoff=0)
expa.add_command(label=f'Alfa: {0}', command=lambda: (write_image(FP_Exp_Acl(imagen, args=0)), show_confirmacion1()))
expa.add_command(label=f'Alfa: {51}', command=lambda: (write_image(FP_Exp_Acl(imagen, args=51)), show_confirmacion1()))
expa.add_command(label=f'Alfa: {102}', command=lambda: (write_image(FP_Exp_Acl(imagen, args=102)), show_confirmacion1()))
expa.add_command(label=f'Alfa: {153}', command=lambda: (write_image(FP_Exp_Acl(imagen, args=153)), show_confirmacion1()))
expa.add_command(label=f'Alfa: {204}', command=lambda: (write_image(FP_Exp_Acl(imagen, args=204)), show_confirmacion1()))



expo = Menu(filtro_puntual, tearoff=0)
expo.add_command(label=f'Alfa: {1}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=1)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {2}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=2)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {3}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=3)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {4}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=4)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {5}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=5)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {6}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=6)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {7}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=7)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {8}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=8)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {9}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=9)), show_confirmacion1()))
expo.add_command(label=f'Alfa: {10}', command=lambda: (write_image(FP_Exp_Osc(imagen, args=10)), show_confirmacion1()))


filtro_puntual.add_cascade(label="Gamma", menu=gamma)
filtro_puntual.add_command(label="Rango Dinámico", command=lambda: (write_image(FP_Log(imagen)), show_confirmacion1()))
filtro_puntual.add_cascade(label="Rango Dinámico parametrizado", menu=logaritmico)
filtro_puntual.add_command(label="Seno", command=lambda: (write_image(FP_Seno(imagen)), show_confirmacion1()))
filtro_puntual.add_command(label="Coseno", command=lambda: (write_image(FP_Coseno(imagen)), show_confirmacion1()))

filtro_puntual.add_cascade(label="Oscuresimiento Exponencial", menu=expo)
filtro_puntual.add_cascade(label="Aclarado Exponencial", menu=expa)
filtro_puntual.add_cascade(label="Binario", menu=binario)
filtro_puntual.add_cascade(label="Binario Inverso", menu=binario_inverso)

filtro_puntual.add_command(label="Sigmoidal Seno", command=lambda: (write_image(FP_Sigmoid_Sin(imagen)), show_confirmacion1()))
filtro_puntual.add_command(label="Sigmoidal Tanh", command=lambda: (write_image(FP_Sigmoid_Tanh(imagen)), show_confirmacion1()))

equal_hist = Menu(filtro_puntual, tearoff=0)


def esp_hist(imagen):
    resultados = histogram_equalization(imagen)
    write_image(resultados[0], 'imagen_original.png')
    write_histogram(resultados[1], 'histograma_original.png')
    write_image(resultados[2], 'imagen_equalizada.png')
    write_histogram(resultados[3], 'histograma_equalizado.png')


equal_hist.add_command(label="Ecualizar Histograma", command=lambda: (esp_hist(imagen), show_confirmacion1()))

filt_spacial = Menu(menu_principal, tearoff=0)
filt_spacial.add_command(label=mask_desc[0], command=lambda: (write_image(Conv(imagen, mask_list[0])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[1], command=lambda: (write_image(Conv(imagen, mask_list[1])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[2], command=lambda: (write_image(Conv(imagen, mask_list[2])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[3], command=lambda: (write_image(Conv(imagen, mask_list[3])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[4], command=lambda: (write_image(Conv(imagen, mask_list[4])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[5], command=lambda: (write_image(Conv(imagen, mask_list[5])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[6], command=lambda: (write_image(Conv(imagen, mask_list[6])), show_confirmacion1()))
filt_spacial.add_command(label=mask_desc[7], command=lambda: (write_image(Conv(imagen, mask_list[7])), show_confirmacion1()))


edges = Menu(filt_spacial, tearoff=0)

def sobel(imagen):
    resultados = detect_edges_sobel(imagen)
    write_image(resultados[0], 'cambios_x.png')
    write_image(resultados[1], 'cambios_y.png')
    write_image(resultados[2], 'magnitud.png')


edges.add_command(label="Sobel", command=lambda: (sobel(imagen), show_confirmacion1()))


def canny(imagen):
    resultados = detect_edges_canny(imagen)
    write_image(resultados[0], 'canny.png')
    write_image(resultados[1], 'post.png')


edges.add_command(label="Canny", command=lambda: (canny(imagen), show_confirmacion1()))


menu_principal.add_cascade(label="Archivo", menu=archivo)
menu_principal.add_cascade(label="Filtro Puntual", menu=filtro_puntual)
menu_principal.add_cascade(label="Histograma", menu=equal_hist)
menu_principal.add_cascade(label="Filtro Espacial", menu=filt_spacial)
menu_principal.add_cascade(label="Bordes", menu=edges)

root.config(menu=menu_principal)
root.mainloop()


