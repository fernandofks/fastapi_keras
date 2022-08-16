import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast
import time

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[-0.23,  0.59,  0.02,  0.16,  0.66,  0.09],
       [-0.23,  0.59,  0.01,  0.15,  0.66,  0.02],
       [-0.23,  0.59,  0.01,  0.13,  0.66, -0.04],
       [-0.23,  0.59,  0.02,  0.13,  0.66, -0.06],
       [-0.22,  0.6 ,  0.11,  0.13,  0.66, -0.06],
       [-0.24,  0.6 ,  0.31,  0.13,  0.66, -0.06],
       [-0.24,  0.62,  0.38,  0.13,  0.66, -0.05],
       [-0.26,  0.6 ,  0.32,  0.13,  0.66, -0.05],
       [-0.27,  0.59,  0.22,  0.13,  0.66, -0.05],
       [-0.25,  0.6 ,  0.14,  0.13,  0.66, -0.05]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    start_time=time.time()
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,11,6)
    input = input[:,1:,:]
    input=np.diff(input, axis=1)
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


