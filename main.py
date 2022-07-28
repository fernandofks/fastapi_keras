import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import ast

MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()
input = np.array([[[0.08, 0.14, 0.09, 0.18],
       [0.08, 0.14, 0.09, 0.22],
       [0.08, 0.14, 0.08, 0.17],
       [0.08, 0.14, 0.08, 0.1 ],
       [0.08, 0.14, 0.07, 0.08],
       [0.08, 0.14, 0.07, 0.08],
       [0.09, 0.2 , 0.07, 0.08],
       [0.1 , 0.32, 0.07, 0.07],
       [0.1 , 0.29, 0.07, 0.07],
       [0.09, 0.24, 0.07, 0.07]]])
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
    return {"prediction": float(prediction)}

# @app.post('/predict/') 
# async def predict(UserInput: UserInput):
#     global input
#     lista_string=UserInput.user_input
#     lista= ast.literal_eval(lista_string)
#     input = np.insert(input[0], 0,lista).reshape(1,11,4)
#     input = input[:,:-1,:]
#     prediction = MODEL.predict(input)
#     return {"prediction": float(prediction)}


