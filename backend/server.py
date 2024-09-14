from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
import base64
import re
# from RAG import stream_llm_answer
app = FastAPI()


# ----- Req. redundant imports

import os
import glob
from collections import deque
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from langchain_community.llms import Ollama
# Import the required message classes
from langchain.schema import HumanMessage, AIMessage
import torchvision
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from RAG import ResNet9

def load_model_with_class():
    # model = torch.load("C:\\banana\\codeables\\Projects\\sih-2024\\backend\\plant-disease-model-complete.pth")
    model = torch.load("./plant-disease-model-complete.pth")
    return model

model = load_model_with_class()
classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Corn_(maize)___healthy',
           'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach___healthy',
           'Pepper,_bell___Bacterial_spot',
           'Pepper,_bell___healthy',
           'Potato___Early_blight',
           'Potato___Late_blight',
           'Potato___healthy',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch',
           'Strawberry___healthy',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___healthy']

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def predict_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output = model(image)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()


# Initialize the LLM
llm = Ollama(model="llama3.1:8b")

# Initialize the tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2")
tokenizer.clean_up_tokenization_spaces = False

# Load documents
loader = DirectoryLoader(
    "C:\\banana\\codeables\\Projects\\sih-2024\\backend\\",
    glob="./Plant diseases.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create or load the FAISS vector store
vectordb = FAISS.load_local(
    folder_path="C:\\banana\\codeables\\Projects\\sih-2024\\backend\\vectordb",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

conversation_history = deque(maxlen=15)

prompt_template = """
You are a chatbot designed to assist users in diagnosing plant diseases. Your task is to help users identify potential issues with their plants based on the symptoms they describe and the images they provide which will be classified using another classifier assume its acuuracy is 100%. 
{context}

**Question: {question}**

**Answer:**
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3, "search_type": "similarity"})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=False
)


def stream_llm_answer(query, img=None, _print=True):
    results = {}
    if img != None:
        imageOut = predict_image(img, model)
        query = query + \
            f".An image was passed to an image model and it classified it as {classes[imageOut]} assume the classifiers acuuracy is 100%. And dont mention the classifier in your answer."
        if _print:
            print(f"predicted class:  {classes[imageOut]}")
        results["label"] = classes[imageOut]
    conversation_history.append(query)
    context = "\n".join(conversation_history)
    formatted_prompt = PROMPT.format(context=context, question=query)

    # Stream response directly from the LLM
    response_stream = llm.stream(formatted_prompt)
    if _print:
        print("llm: ", end="")

    final_response = ""
    for chunk in response_stream:
        if _print:
            print(chunk, end="", flush=True)
        final_response += chunk

    if _print:
        print()  # Newline after response is complete
    conversation_history.append(final_response)
    results["description"] = final_response
    return results

# ----- End. Redundant imports

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
async def detect_base64(req: Request):
    reqJson = await req.json()
    imgB64 = reqJson["image"]
    print(imgB64[:30])
    image = re.sub('^data:image/.+;base64,', '', imgB64)
    with open("cur_image.png", "wb") as f:
        f.write(base64.b64decode(image + "=="))

    res = stream_llm_answer(
        "Detect the disease in the image", img="./cur_image.png", _print=False)
    return res
    # return {"label": "Apple Scab", "confidence": 96, "description": "# Apple Scab\nApple Scab is a disease caused by the fungus Venturia inaequalis. It is a common disease of apple and crabapple trees, as well as mountain ash and pear. The disease manifests as dull black or gray-brown lesions on the leaves, fruit, and twigs of the tree. The lesions may have a velvety appearance and can cause the leaves to become distorted or drop prematurely. Apple Scab can reduce the quality and yield of fruit, and severe infections can weaken the tree and make it more susceptible to other diseases. The disease is most severe in wet, humid conditions, and can be managed through cultural practices, such as pruning and sanitation, as well as fungicide applications."}
