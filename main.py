import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[ 2.66, -1.94,  0.66, -0.07],
       [ 2.66, -1.94,  0.66, -0.07],
       [ 2.65, -2.1 ,  0.66, -0.07],
       [ 2.65, -2.2 ,  0.66, -0.07],
       [ 2.65, -2.22,  0.66, -0.07],
       [ 2.65, -2.22,  0.68,  0.01],
       [ 2.64, -2.22,  0.68,  0.18],
       [ 2.64, -2.22,  0.69,  0.26],
       [ 2.64, -2.22,  0.67,  0.16],
       [ 2.64, -2.22,  0.67,  0.03]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,11,4)
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


