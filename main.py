import uvicorn
import numpy as np
import cv2 as cv
import tensorflow.keras as keras
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import io
import base64

app = FastAPI()

CLASSES = {0: 'angry',
           1: 'disgust',
           2: 'fear',
           3: 'happy',
           4: 'neutral',
           5: 'sad',
           6: 'surprise'}


class Item(BaseModel):
    img: str


model = keras.models.load_model('my_arch64.h5')


async def FetchFace(img):
    img = np.array(img)
    fc = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    detection_result, rejectLevels, levelWeights = fc.detectMultiScale3(img, outputRejectLevels=True,
                                                                        scaleFactor=1.1, minNeighbors=3,
                                                                        minSize=(48, 48))
    try:
        levelWeights = levelWeights.reshape(len(levelWeights))
        max_index = list(levelWeights).index(levelWeights.max())
        x, y, w, h = detection_result[max_index]

        face = img[y:y + h, x:x + w]
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        return cv.resize(gray, (128, 128))
    except:
        return -1


@app.post('/')
async def GetMood(item: Item):
    b = base64.b64decode(item.img)
    img = np.array(Image.open(io.BytesIO(b)))

    fc = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    detection_result, rejectLevels, levelWeights = fc.detectMultiScale3(img, outputRejectLevels=True,
                                                                        scaleFactor=1.1, minNeighbors=3,
                                                                        minSize=(48, 48))
    try:
        levelWeights = levelWeights.reshape(len(levelWeights))
        max_index = list(levelWeights).index(levelWeights.max())
        x, y, w, h = detection_result[max_index]

        face = img[y:y + h, x:x + w]
        gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        img = cv.resize(gray, (128, 128))
    except:
        return "Couldn't detect a face"

    input_arr = np.array([img])
    predictions = model.predict(input_arr)
    predictions = list(predictions.reshape(7))
    prediction = CLASSES[predictions.index(max(predictions))]
    return prediction


@app.get('/')
async def Welcome():
    return 'Welcome to Mood Classifier API'


