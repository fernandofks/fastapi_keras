import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast
import time

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[-0.1 ,  0.62,  0.78,  0.04,  0.01,  0.7 ,  0.71,  0.04],
       [ 0.06, -0.76, -0.64,  0.02,  0.02,  0.7 ,  0.71,  0.05],
       [ 0.05, -0.75, -0.66,  0.03,  0.02,  0.7 ,  0.71,  0.04],
       [ 0.01, -0.72, -0.69,  0.02,  0.01,  0.69,  0.72,  0.04],
       [ 0.03, -0.71, -0.71,  0.01,  0.01,  0.69,  0.72,  0.04],
       [ 0.04, -0.69, -0.73,  0.02,  0.02,  0.69,  0.73,  0.03],
       [ 0.04, -0.69, -0.73,  0.  ,  0.03,  0.63,  0.78,  0.02],
       [ 0.04, -0.68, -0.73,  0.  , -0.12, -0.53, -0.84,  0.02],
       [-0.04,  0.68,  0.73,  0.  , -0.02,  0.72,  0.7 ,  0.04],
       [-0.05,  0.68,  0.73,  0.  , -0.12,  0.74,  0.66,  0.07]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    start_time=time.time()
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,11,8)
    input = input[:,1:,:]
    prediction = MODEL.predict(input)
    vel = np.argmax(prediction)
    print(input)
    process_time = time.time() - start_time
    print(process_time)
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


