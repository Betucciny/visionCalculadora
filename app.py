from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from funciones import *
from tkinter import filedialog as fd
from functools import partial
from PIL import ImageTk, Image


def change_image(filename, label1, label2):
    new_image = ImageTk.PhotoImage(Image.open(filename).resize((400, 300)))
    label1.configure(image=new_image)
    label1.image = new_image
    label2.configure(image=new_image)
    label2.image = new_image
    root.update()


def select_file(label1, label2):
    filetypes = (
        ('Imagenes', '*.png *.jpg *.jpeg *.gif'),
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    try:
        change_image(filename, label1, label2)
        global imageToProcess
        imageToProcess = cv2.imread(filename)
        imageToProcess = cv2.resize(imageToProcess, (400, 300))
    except:
        print(f"No se ha seleccionado ninguna imagen {filename}")
        pass


def apply_filter(select: str, original, aplied, arg):
    func, title, args = list_func[select]
    imgE = cv2.cvtColor(func(original, arg), cv2.COLOR_BGR2RGB)
    cv2.imwrite('{}.png'.format(title), imgE)
    image = Image.fromarray(imgE, 'RGB')
    image = ImageTk.PhotoImage(image=image)
    aplied.configure(image=image)
    aplied.image = image
    root.update()


root = Tk()
root.minsize(900, 450)
root.maxsize(900, 450)
root.title("Calculadora de imagenes")
root.geometry("900x450")

top = Frame(root)
bottom = Frame(root)
top.pack(side=TOP)
bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

imageToProcess = cv2.imread('rumbling.png')
imageToProcess = cv2.resize(imageToProcess, (400, 300))

title = Label(root, text="Calculadora de imagenes", font=("Arial", 24))
title.pack(in_=top, side=TOP, padx=10, pady=10)

image = ImageTk.PhotoImage(Image.open('rumbling.png').resize((400, 300)))
label_image_original = Label(root, image=image, width=400, height=300)
label_image_original.pack(in_=top, side=LEFT, padx=10, pady=10)

label_image_processed = Label(root, image=image, width=400, height=300)
label_image_processed.pack(in_=top, side=RIGHT, padx=10, pady=10)


buttonChange = Button(root, text="Cambiar imagen", command=partial(select_file, label_image_original, label_image_processed))
buttonChange.pack(in_=bottom, side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)

opciones = [i for i in list_func.keys()]
clicked = StringVar()
clicked.set(opciones[0])

buttonApply = Button(root, text="Aplicar", command=lambda : apply_filter(clicked.get(), imageToProcess, label_image_processed, float(opciones_arg_dict[clicked_arg.get()])))
buttonApply.pack(in_=bottom, side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)

drop = OptionMenu(root, clicked, *opciones)
drop.pack(in_=bottom, side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)

opciones_arg_dict = {"1/5": 1/5, "1/4": 1/4, "1/3": 1/3, "1/2": 1/2, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
clicked_arg = StringVar()
opciones_arg = [i for i in opciones_arg_dict.keys()]
clicked_arg.set(opciones_arg[0])

drop_arg = OptionMenu(root, clicked_arg, *opciones_arg)
drop_arg.pack(in_=bottom, side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)




root.mainloop()


