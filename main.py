from fastapi import FastAPI

# Code for creating the model of the API
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Neural Netwrok libraries
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense

# Creating std scaler
sc = StandardScaler()

# Loading the model saved
model = keras.models.load_model("Test_model_Neural_Network")

# List for prediction
pred_label = list()

# Defining data preparation
def preparazione(dati_originali):
    dati_pronti = dati_originali.values
    dati_pronti = sc.fit_transform(dati_originali)
    return dati_pronti

# Defining function for prediction
def predict(dati):
    previsione = model.predict(dati)
    return previsione

# Defining method for converting results
def convert(dati_preparati, lista):
    for i in range(len(dati_preparati)):
        result = pred_label.append(np.argmax(dati_preparati[i]))

    return result

app = FastAPI()


@app.post("/prediciton")
def get_prediction():
    dati_preparati = preparazione(Dati)
    prediction_heart = predict(dati_preparati)
    prediction_label = convert(prediction_heart, pred_label)
    return prediction_label
