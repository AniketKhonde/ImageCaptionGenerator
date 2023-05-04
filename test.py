# Importing libraries
import sys, time, os, warnings
from keras.preprocessing.sequence import pad_sequences

dir_Flickr_jpg = "C:/Users/suraj/PycharmProjects/ImageCaptionGenerator/Flicker8k_Dataset/"
## The location of the caption file
dir_Flickr_text = "C:/Users/suraj/Downloads/Flickr8k.token.txt"

jpgs = os.listdir(dir_Flickr_jpg)
print("The number of jpg flies in Flicker8k: {}".format(len(jpgs)))
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from collections import Counter
from keras.applications.vgg16 import preprocess_input
model1 = tf.keras.models.load_model('model_adv.h5')
warnings.filterwarnings("ignore")
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from copy import copy
## read in the Flickr caption data
file = open(dir_Flickr_text,'r')
text = file.read()
file.close()
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict

modelvgg = tf.keras.applications.VGG16(include_top=True, weights=None)
## load the locally saved weights
modelvgg.load_weights("C:/Users/suraj/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.summary()
images = OrderedDict()
npix = 224
target_size = (npix, npix, 3)
data = np.zeros((len(jpgs), npix, npix, 3))
print("Complete")

for i, name in enumerate(jpgs):
    # load an image from file
    filename = dir_Flickr_jpg + '/' + name
    image = load_img(filename, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    nimage = preprocess_input(image)

    y_pred = modelvgg.predict(nimage.reshape((1,) + nimage.shape[:3]))
    images[name] = y_pred.flatten()


datatxt = []
for line in text.split('\n'):
    col = line.split('\t')
    if len(col) == 1:
        continue
    w = col[0].split("#")
    datatxt.append(w + [col[1].lower()])

df_txt = pd.DataFrame(datatxt,columns=["filename","index","caption"])


uni_filenames = np.unique(df_txt.filename.values)
print("The number of unique file names : {}".format(len(uni_filenames)))
print("The distribution of the number of captions for each image:")

# Counting number of captions for each image using counter
Counter(Counter(df_txt.filename.values).values())
def add_start_end_seq_token(captions):
    caps = []
    for txt in captions:
        txt = 'startseq ' + txt + ' endseq'
        caps.append(txt)
    return(caps)
df_txt0 = copy(df_txt)
df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])
df_txt0.head(5)
dimages, keepindex = [], []
df_txt0 = df_txt0.loc[df_txt0["index"].values == "0", :]
for i, fnm in enumerate(df_txt0.filename):
    if fnm in images.keys():
        dimages.append(images[fnm])
        keepindex.append(i)

fnames = df_txt0["filename"].iloc[keepindex].values
dcaptions = df_txt0["caption"].iloc[keepindex].values
dimages = np.array(dimages)

from keras.preprocessing.text import Tokenizer
## the maximum number of words in dictionary
nb_words = 8000
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(dcaptions)
vocab_size = len(tokenizer.word_index) + 1
print("vocabulary size : {}".format(vocab_size))
dtexts = tokenizer.texts_to_sequences(dcaptions)
import os
from keras.preprocessing.image import load_img, img_to_array


os.chdir('C:\\Users\\suraj\\desktop\\')
os.getcwd()

index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])

def predict_caption(image):
    '''
    image.shape = (1,4462)
    '''

    in_text = 'startseq'

    for iword in range(30):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],30)
        yhat = model1.predict([image,sequence],verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    return(in_text)



def upload_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                            filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 300
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) - 1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def caption():
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    nimage = preprocess_input(numpy_image)

    feature = modelvgg.predict(nimage.reshape((1,) + nimage.shape[:3]))
    caption = predict_caption(feature)
    table = tk.Label(frame, text="Caption: " + caption[9:-7], font=("Helvetica", 12)).pack()