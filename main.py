import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[274.86, 256.81, 244.33, 272.67, 111.92,  76.86],
       [274.89, 259.44, 243.97, 272.43, 113.23,  75.46],
       [275.5 , 318.13, 199.05, 272.4 , 118.4 ,  70.28],
       [279.01, 348.03, 182.32, 272.84, 119.84,  68.54],
       [271.82, 247.06, 280.01, 272.64, 120.84,  67.66],
       [279.06, 217.48, 313.49, 272.87, 119.44,  69.24],
       [272.24, 243.18, 298.52, 272.72, 116.63,  71.96],
       [273.44, 267.7 , 272.8 , 273.12, 115.59,  73.05],
       [280.89, 349.12, 196.16, 273.14, 114.45,  74.14],
       [283.84, 351.37, 189.33, 273.18, 120.4 ,  68.28]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,11,6)
    input = input[:,1:,:]
    prediction = MODEL.predict(input)
    vel = np.argmax(prediction)
    print(vel)
    return {"prediction": float(vel/2)}

# @app.post('/predict/') 
# async def predict(UserInput: UserInput):
#     global input
#     lista_string=UserInput.user_input
#     lista= ast.literal_eval(lista_string)
#     input = np.insert(input[0], 0,lista).reshape(1,11,4)
#     input = input[:,:-1,:]
#     prediction = MODEL.predict(input)
#     return {"prediction": float(prediction)}


