import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from keras.models import load_model
import PIL.ImageOps as ImageOps
model = tf.keras.models.load_model('new_model.h5')
data = np.ndarray(shape=(1, 256, 256,3), dtype=np.float32)
class_names = ['adho mukha svanasana',
                'adho mukha vriksasana',
                'agnistambhasana',
                'ananda balasana',
                'anantasana',
                'anjaneyasana',
                'ardha bhekasana',
                'ardha chandrasana',
                'ardha matsyendrasana',
                'ardha pincha mayurasana',
                'ardha uttanasana',
                'ashtanga namaskara',
                'astavakrasana',
                'baddha konasana',
                'bakasana',


               ]

top = tk.Tk()
top.geometry('800x600')
top.title('Yoga Pose Detection')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    test_image = image.resize((256, 256))
    test_image = np.array(test_image)
    test_image = test_image / 256.0
    test_image = test_image[:, :, :3]
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)

    pred_label = np.argmax(prediction)  
    sign = class_names[pred_label]
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Detect!", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Yoga Pose Detection", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()