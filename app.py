import json
from io import BytesIO
import cv2
import os
from os.path import join, isdir, isfile
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
import random

import tensorflow as tf

st.set_page_config(layout='wide')

@st.cache()
def load_model(path='models/v1_efficientnetb0/fine_tuned_model'):
    """Retrieves the trained model"""
    return tf.keras.models.load_model(path)

@st.cache()
def get_class_names_and_shape(path='models/v1_efficientnetb0/data.json'):
    with open(path, 'r') as fl:
        data = json.load(fl)
    
    return data['class_names'], data['image_shape']

@st.cache()
def predict(img, class_names, model):
    pred = model.predict(tf.expand_dims(img, axis=0)).squeeze()
    pred_df = pd.DataFrame()
    pred_df['Flower Name'] = [i.title() for i in class_names]
    pred_df['Confident Level'] = np.round(pred*100, 2)
    pred_df.sort_values('Confident Level', ascending=False, inplace=True)
    pred_df['Confident Level'] = ["{:.2f}%".format(i) for i in pred_df['Confident Level']]
    pred_df = pred_df.reset_index(drop=True)
    pred_df.index = pred_df.index + 1
    return pred_df

def image_to_numpy(file_dir, image_size=None):
    img =  cv2.cvtColor(cv2.imread(file_dir), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, image_size) if image_size else img

def get_dirs(path):
    return [name for name in os.listdir(path) if isdir(join(path, name))]

def get_files(path):
    return [name for name in os.listdir(path) if isfile(join(path, name))]

def get_random_imgs(data_dir, rand_imgs=5, equal_img_per_class=None, rand_classes=None):
    data_dir = str(data_dir)
    class_names = get_dirs(data_dir)

    if rand_classes:
        for class_name in rand_classes:
            if class_name not in class_names:
                raise ValueError(f'"{class_name}" not found in "{data_dir}""')
    else:
        rand_classes = class_names

    if equal_img_per_class:
        rand_list = {class_name : equal_img_per_class for class_name in rand_classes}
    else:
        rand_list = {class_name : 0 for class_name in rand_classes}
        for class_name in random.choices(rand_classes, k=rand_imgs):
            rand_list[class_name] += 1

    rand = []
    for class_name, rand_img_num in rand_list.items():
        if rand_img_num:
            class_dir = join(data_dir, class_name)
            rand_images = random.choices(get_files(class_dir), k=rand_img_num)
            for i in rand_images:
                rand.append(join(class_dir, i))
    return rand

if __name__ == '__main__':
    model = load_model()
    class_names, IMAGE_SHAPE = get_class_names_and_shape()
    num_classes = len(class_names)

    st.title('Welcome To Mini Flower!')
    instructions = f"""
        Here, you can classify {num_classes} types of Flowers.
        These are : {', '.join(sorted(class_names))}
        """
    st.write(instructions)
    file = st.file_uploader('Upload An Image of Flower')
    
    if file:
        img = Image.open(file)
        prediction = predict(np.array(img.resize(IMAGE_SHAPE)), class_names, model)
        img = np.array(img)
    else:
        img = get_random_imgs('samples', rand_imgs=1)
        img = Image.open(img[0])
        prediction = predict(np.array(img.resize(IMAGE_SHAPE)), class_names, model)
        img = np.array(img)

    st.title("Here is the image you've selected")
    st.image(img)
    st.title("Here are the five most likely flowers")
    st.write(prediction.to_html(escape=False), unsafe_allow_html=True)
    st.title(f"Here are some images of {prediction.iloc[0, 0]}")

    imgs = get_random_imgs('samples', rand_imgs=3, rand_classes=[prediction.iloc[0, 0].lower()])
    st.image([Image.open(img) for img in imgs])
