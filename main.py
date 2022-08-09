import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[ 0.17,  0.13,  0.75, -0.26,  0.07,  0.08,  4.75,  0.02],
       [ 0.12,  0.13,  0.75, -0.26,  0.07,  0.06,  4.75,  0.04],
       [ 0.16,  0.14,  0.76, -0.26,  0.07,  0.02,  4.75,  0.06],
       [ 0.11,  0.13,  0.76, -0.26,  0.07, -0.01,  4.75,  0.06],
       [ 0.11,  0.14,  0.76, -0.26,  0.07, -0.05,  4.76,  0.05],
       [ 0.12,  0.14,  0.76, -0.26,  0.07, -0.09,  4.76,  0.03],
       [ 0.14,  0.14,  0.76, -0.26,  0.07, -0.13,  4.76,  0.02],
       [ 0.18,  0.14,  0.76, -0.26,  0.07, -0.15,  4.76, -0.01],
       [ 0.17,  0.14,  0.76, -0.25,  0.08, -0.15,  4.76, -0.02],
       [ 0.17,  0.13,  0.77, -0.21,  0.07, -0.13,  4.76, -0.02]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,11,8)
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


