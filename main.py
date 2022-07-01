import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel

import numpy 

class TypedArray(numpy.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return numpy.array(val, dtype=cls.inner_type)

class ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (TypedArray,), {'inner_type': t})

class Array(numpy.ndarray, metaclass=ArrayMeta):
    pass

MODEL = tf.keras.models.load_model('model/')

app = FastAPI()

class UserInput(BaseModel):
    user_input: Array[float]

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(UserInput: UserInput):

    prediction = MODEL.predict([UserInput.user_input])

    return {"prediction": float(prediction)}