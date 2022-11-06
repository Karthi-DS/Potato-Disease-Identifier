import io
import PIL
import numpy as np
from fastapi import FastAPI,UploadFile,File
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI()

CLASS_NAMES = ['Early_blight','Late_blight','Healthy']
model = tf.keras.models.load_model(r"C:\Users\KARTHEESVARAN\OneDrive\Desktop\potato disease\10")

@app.get('/live')
async def live():
    return 'You are live'

def read_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')

async def post(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    img = read_as_image(bytes)
    c_image = np.expand_dims(img,0)
    pred  = model.predict(c_image)
    prediction = CLASS_NAMES[np.argmax(pred[0])]
    pre = prediction
    return pre



if __name__ == '__main__':
    uvicorn.run(app,host = 'localhost',port = 7000)