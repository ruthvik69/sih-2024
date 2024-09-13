from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64
# from RAG import stream_llm_answer

app = FastAPI()

origins = [
    "http://localhost:8081",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Input(BaseModel):
    img: UploadFile


@app.post("/detect-image")
def detect(image: UploadFile = File(...)):
    with open("cur_image.png", "wb") as f:
        f.write(image.file.read())

    # res = stream_llm_answer(
    #     "Detect the disease in the image", img="./cur_image.png", _print=False)
    # return res
    return {"label": "Apple Scab", "confidence": 96, "description": "# Apple Scab\nApple Scab is a disease caused by the fungus Venturia inaequalis. It is a common disease of apple and crabapple trees, as well as mountain ash and pear. The disease manifests as dull black or gray-brown lesions on the leaves, fruit, and twigs of the tree. The lesions may have a velvety appearance and can cause the leaves to become distorted or drop prematurely. Apple Scab can reduce the quality and yield of fruit, and severe infections can weaken the tree and make it more susceptible to other diseases. The disease is most severe in wet, humid conditions, and can be managed through cultural practices, such as pruning and sanitation, as well as fungicide applications."}


@app.post("/detect")
def detect_base64(image: str):
    with open("cur_image.png", "wb") as f:
        f.write(base64.b64decode(image))

    # res = stream_llm_answer(
    #     "Detect the disease in the image", img="./cur_image.png", _print=False)
    # return res
    return {"label": "Apple Scab", "confidence": 96, "description": "# Apple Scab\nApple Scab is a disease caused by the fungus Venturia inaequalis. It is a common disease of apple and crabapple trees, as well as mountain ash and pear. The disease manifests as dull black or gray-brown lesions on the leaves, fruit, and twigs of the tree. The lesions may have a velvety appearance and can cause the leaves to become distorted or drop prematurely. Apple Scab can reduce the quality and yield of fruit, and severe infections can weaken the tree and make it more susceptible to other diseases. The disease is most severe in wet, humid conditions, and can be managed through cultural practices, such as pruning and sanitation, as well as fungicide applications."}
