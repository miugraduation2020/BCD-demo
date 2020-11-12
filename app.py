from tkinter import *
from string import ascii_uppercase
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
from keras.models import load_model
import cv2


class App():
    def __init__(self, master=None):
        self.img = ""
        self.master = master
        self.panel = Label()
        self.letter = Label()
        self.output = None
        self.create_widgets()

    def create_widgets(self):
        title = Label(self.master, text="Breast Cancer Type Prediction", font=(
            "Helvetica", 30, "bold"))
        subtitle = Label(self.master, text="Cancer Prediction",
                         font=("Helvetica", 22))
        l1 = Label(self.master, height=1)
        btn = Button(self.master, text='open image', command=self.open_img)
        l2 = Label(self.master, height=1)
        title.pack()
        subtitle.pack()
        l1.pack()
        btn.pack()
        l2.pack()

    def open_img(self):
        x = self.openfilename()
        img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 784)
        img = img / 255

        model = load_model("model.h5")
        self.output = model.predict(img)
        self.output = np.argmax(self.output)

        self.img = Image.open(x)
        self.img = self.img.resize((250, 250), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)

        self.panel['image'] = self.img
        self.panel['width'] = 250
        self.panel['height'] = 250
        self.panel['relief'] = "solid"
        self.panel["bd"] = 1
        self.panel.pack()

        l1 = Label(self.master, height=1)
        l1.pack()

        tummor = " "
        LETTERS = {letter: str(index) for index,
                   letter in enumerate(ascii_uppercase, start=0)}
        keys = list(LETTERS.keys())
        values = list(LETTERS.values())
        if(keys[values.index(str(self.output))] == "A"):
            tummor = "Benign"
        elif(keys[values.index(str(self.output))] == "B"):
            tummor = "Malignant"
        self.letter["text"] = "Predicted Tummor: " + tummor

        self.letter["font"] = ("Helvetica", 30)
        self.letter.pack()

    def openfilename(self):
        filename = filedialog.askopenfilename(title='"pen')
        return filename


root = Tk()
root.title("Image Loader")
root.geometry("700x500")
root.resizable(width=True, height=True)

app = App(root)

root.mainloop()
