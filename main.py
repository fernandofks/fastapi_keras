import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[ 0.25,  0.19,  0.76,  0.03, -0.73,  0.16, -0.05,  4.59,  0.07],
       [ 0.27,  0.19,  0.76,  0.03, -0.73,  0.16, -0.06,  4.59,  0.06],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.07,  4.59,  0.06],
       [ 0.31,  0.19,  0.76,  0.03, -0.73,  0.16, -0.08,  4.59,  0.06],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.59,  0.05],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.59,  0.05],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.59,  0.05],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.6 ,  0.05],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.6 ,  0.05],
       [ 0.32,  0.19,  0.76,  0.03, -0.73,  0.16, -0.09,  4.6 ,  0.05]]])
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


