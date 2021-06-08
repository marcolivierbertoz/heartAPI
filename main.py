# New Version of the API, this time with training the model
# and not loading it.
# Model used is the K-NN, not anymore the Tensorflow
# There will be only training, not testng the model
# The aim of the API is to learn how to create and use one, not creating a perfect model for predicting



# Importing packages ##############################################################################
# FastAPI for creating API
import uvicorn
from fastapi import FastAPI

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# Python Package
import numpy as np
import pandas as pd

# Download basic ML model for creating class
from pydantic import BaseModel

# Sci-kit Learn Packages, for creating Machine Learning models ######
# Package for standardizing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Train Test Split
# from sklearn.model_selection import train_test_split

# Model K-NN
from sklearn.neighbors import KNeighborsClassifier

#####################################################################################################

# Model training ##############################################################################

# Loading data
heart_original = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Retrving the FEatures and Target
X_features = heart_original.iloc[:,:12].values
y_target = heart_original.iloc[:,12:13].values

# Normalizing the data
sc = StandardScaler()
One_Hot = OneHotEncoder()

X_features_normalized = sc.fit_transform(X_features)

y_target_encoded = One_Hot.fit_transform(y_target).toarray()
# Creating the model
knn = KNeighborsClassifier(n_neighbors=10, algorithm='brute', n_jobs=-1)

# Training the model
knn.fit(X_features_normalized,y_target_encoded)


#################################################################################################

# Defining fucntion for elaborate inout data form webapp ###################################

# Data preparation
def preparazione(dati_originali):
    dati_pronti = sc.fit_transform(dati_originali)
    return dati_pronti

# Predicitng the results
def predict(dati):
    previsione = knn.predict(dati)
    return previsione




##################################################################################################


# Creating API ################################################################################

# Creating class for validation
class Heart(BaseModel):
    Age: float
    Anemia: int
    CPK: int
    Diabetes: int
    Ejection_fraction: int
    HBP: int
    Platelets: float
    Serum_Creatinine: float
    Serum_Sodium: int
    Woman_Man: int
    Time: int

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Welcome to my first API!"}

@app.post("/prediction")
async def get_prediction(heart :Heart):
    json_compatible_data = jsonable_encoder(heart)
    input_df = pd.DataFrame(json_compatible_data)


    # input_df = pd.DataFrame([heart.dict()])
    dati_preparati = preparazione(input_df)
    prediction_heart = predict(dati_preparati)
    #prediction_label = convert(prediction_heart, pred_label)
    #return JSONResponse(content=prediction_label)
    return prediction_heart

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)  
#####################################################################################################