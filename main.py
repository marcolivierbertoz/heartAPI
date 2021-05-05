import uvicorn
from fastapi import FastAPI

# Downlaoding encoder
from fastapi.encoders import jsonable_encoder

# Download basic ML model for creating class
from pydantic import BaseModel

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


# Creating class for validation
class Heart(BaseModel):
    age: float
    anemia: int
    cpk: int
    diabetes: int
    eject_fraction: int
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: int
    sex: int
    time: int

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

@app.get("/")
def home():
    return {"message":"Welcome to my first API!"}

@app.post("/prediction")
async def get_prediction(heart :Heart):
    heart_direct = jsonable_encoder(heart)
    for key, value in heart_direct.items():
        heart_direct[key] = [value]

    input_df = pd.DataFrame.from_dict(heart_direct)    
    #input_df = pd.DataFrame(heart)
    # input_df = pd.DataFrame([heart.dict()])
    dati_preparati = preparazione(input_df)
    prediction_heart = predict(dati_preparati)
    prediction_label = convert(prediction_heart, pred_label)
    return prediction_label

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)    


