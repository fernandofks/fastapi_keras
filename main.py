import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast
import time

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
       [-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
       [-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
       [-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
    [-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
        [-0.198,  0.629,  0.259,  0.094,  0.628,  0.307],
       [-0.198,  0.628,  0.259,  0.094,  0.628,  0.307]]])
# class UserInput(BaseModel):
#     user_input: str
@app.get('/{UserInput}')
async def predicting(UserInput):
    start_time=time.time()
    global input
    lista_string=UserInput
    lista= ast.literal_eval(lista_string)
    input = np.append(input[0],[lista],axis=0).reshape(1,8,6)
    input = input[:,1:,:]
    input_diff=np.diff(input, axis=1)
    prediction = MODEL.predict(input_diff)
    vel = np.argmax(prediction, axis=1)
    process_time = time.time() - start_time
    print(process_time)
    return {"prediction": float(vel)}

# @app.post('/predict/') 
# async def predict(UserInput: UserInput):
#     global input
#     lista_string=UserInput.user_input
#     lista= ast.literal_eval(lista_string)
#     input = np.insert(input[0], 0,lista).reshape(1,11,4)
#     input = input[:,:-1,:]
#     prediction = MODEL.predict(input)
#     return {"prediction": float(prediction)}


