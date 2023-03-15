from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from funciones import *
from tkinter import filedialog as fd
from functools import partial
from PIL import ImageTk, Image

def abrir_imagen():
    global imagen
    ruta = fd.askopenfilename(title = "Abrir Imagen", filetypes = (("png files","*.png"),("all files","*.*")))
    imagen = cv2.imread(ruta)


imagen = cv2.imread('rumbling.png')

root = Tk()
root.title("Calculadora de Imagenes")
root.geometry("300x20")

menu_principal = Menu(root)


archivo = Menu(menu_principal, tearoff=0)

archivo.add_command(label="Abrir Imagen", command=abrir_imagen)

editar = Menu(menu_principal, tearoff=0)

filtro_puntual = Menu(editar, tearoff=0)

filtro_puntual.add_command(label="Identidad", command=lambda : FP_Iden(imagen))
filtro_puntual.add_command(label="Negativo", command=lambda : FP_Neg(imagen))
filtro_puntual.add_command(label="Grises", command=lambda : FP_Grises(imagen))

binario_threshold = [i for i in range(0, 256, 51)]

binario = Menu(filtro_puntual, tearoff=0)
for i in binario_threshold:
    binario.add_command(label=str(i), command=partial(FP_Bin, imagen, args=i))

binario_inverso = Menu(filtro_puntual, tearoff=0)
for i in binario_threshold:
    binario_inverso.add_command(label=str(i), command=partial(FP_BinInv, imagen, args=i))

logarit_m = [i for i in range(1, 11)]

logaritmico = Menu(filtro_puntual, tearoff=0)
for i in logarit_m:
    logaritmico.add_command(label=str(i), command=partial(FP_Log, imagen, args=i))

gamma_m = [i/2 for i in range(1, 11)]

gamma = Menu(filtro_puntual, tearoff=0)
for i in gamma_m:
    gamma.add_command(label=str(i), command=partial(FP_gamma, imagen, args=i))

filtro_puntual.add_command(label="Seno", command=lambda : FP_Seno(imagen))
filtro_puntual.add_command(label="Coseno", command=lambda : FP_Coseno(imagen))

filtro_puntual.add_cascade(label="Binario", menu=binario)
filtro_puntual.add_cascade(label="Binario Inverso", menu=binario_inverso)
filtro_puntual.add_cascade(label="Logaritmico", menu=logaritmico)
filtro_puntual.add_cascade(label="Gamma", menu=gamma)

editar.add_cascade(label="Filtro Puntual", menu=filtro_puntual)






menu_principal.add_cascade(label="Archivo", menu=archivo)
menu_principal.add_cascade(label="Editar", menu=editar)


# Asignar el menú principal a la ventana principal
root.config(menu=menu_principal)

root.mainloop()


