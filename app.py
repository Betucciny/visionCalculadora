from tkinter import *
from tkinter.messagebox import showinfo
from funciones import *
from tkinter import filedialog as fd
from functools import partial


def abrir_imagen():
    global imagen
    ruta = fd.askopenfilename(title = "Abrir Imagen", filetypes = (("png files","*.png"),("all files","*.*")))
    imagen = cv2.imread(ruta)
    showinfo(title="Confirmacion", message="Imagen Guardada")


def show_confirmacion2():
    showinfo(title="Confirmacion", message="Imagen Editada con exito")


imagen = cv2.imread('rumbling.png')

root = Tk()
root.title("Calculadora de Imagenes")
root.geometry("300x20")

menu_principal = Menu(root)


archivo = Menu(menu_principal, tearoff=0)
archivo.add_command(label="Abrir Imagen", command=abrir_imagen)


filtro_puntual = Menu(menu_principal, tearoff=0)

filtro_puntual.add_command(label="Identidad", command=lambda : (FP_Iden(imagen), show_confirmacion2()))
filtro_puntual.add_command(label="Negativo", command=lambda : (FP_Neg(imagen), show_confirmacion2()))
filtro_puntual.add_command(label="Grises", command=lambda : (FP_Grises(imagen), show_confirmacion2()))

binario_threshold = [i for i in range(0, 256, 51)]

binario = Menu(filtro_puntual, tearoff=0)
for i in binario_threshold:
    binario.add_command(label=f'Umbral: {i}', command=lambda: (FP_Bin(imagen, args=i), show_confirmacion2()))

binario_inverso = Menu(filtro_puntual, tearoff=0)
for i in binario_threshold:
    binario_inverso.add_command(label=f'Umbral: {i}', command=lambda: (FP_BinInv(imagen, args=i), show_confirmacion2()))

logarit_m = [i for i in range(1, 11)]

logaritmico = Menu(filtro_puntual, tearoff=0)
for i in logarit_m:
    logaritmico.add_command(label=f'Alfa: {i}', command=lambda: (FP_Log(imagen, args=i), show_confirmacion2()))

gamma_m = [i/4 for i in range(1, 11)]

gamma = Menu(filtro_puntual, tearoff=0)
for i in gamma_m:
    gamma.add_command(label=f'Gamma: {i}', command=lambda: (FP_gamma(imagen, args=i), show_confirmacion2()))

expa = Menu(filtro_puntual, tearoff=0)
for i in binario_threshold:
    expa.add_command(label=f'Alfa: {i}', command=lambda: (FP_Exp_Acl(imagen, args=i), show_confirmacion2()))


expo_a = [i/4 for i in range(1, 11)]
expo = Menu(filtro_puntual, tearoff=0)
for i in expo_a:
    expo.add_command(label=f'Alfa: {i}', command=lambda: (FP_Exp_Osc(imagen, args=i), show_confirmacion2()))

filtro_puntual.add_cascade(label="Gamma", menu=gamma)
filtro_puntual.add_command(label="Rango Dinámico", command=lambda : (FP_Log(imagen), show_confirmacion2()))
filtro_puntual.add_cascade(label="Rango Dinámico parametrizado", menu=logaritmico)
filtro_puntual.add_command(label="Seno", command=lambda: (FP_Seno(imagen), show_confirmacion2()))
filtro_puntual.add_command(label="Coseno", command=lambda: (FP_Coseno(imagen), show_confirmacion2()))
filtro_puntual.add_cascade(label="Oscuresimiento Exponencial", menu=expo)
filtro_puntual.add_cascade(label="Aclarado Exponencial", menu=expa)


filtro_puntual.add_cascade(label="Binario", menu=binario)
filtro_puntual.add_cascade(label="Binario Inverso", menu=binario_inverso)







menu_principal.add_cascade(label="Archivo", menu=archivo)
menu_principal.add_cascade(label="Filtro Puntual", menu=filtro_puntual)


root.config(menu=menu_principal)
root.mainloop()


