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


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(
        ), nn.Conv2d(128, 128, kernel_size=3, padding=1))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.res2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(
        ), nn.Conv2d(512, 512, kernel_size=3, padding=1))

        self.classifier = nn.Sequential(nn.MaxPool2d(
            4), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, xb):
        out = F.relu(self.conv1(xb))
        out = F.relu(self.conv2(out))
        out = self.res1(out) + out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


modelBase = torch.load("plant-disease-model-complete.pth",
                   map_location=torch.device('cpu'))

torch.save(modelBase.state_dict(), "model_weights.pth")

model = ResNet9(in_channels=512, num_classes=38)  # Initialize the model
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))


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


if __name__ == "__main__":
    class ResNet9(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(ResNet9, self).__init__()

            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.res1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(
            ), nn.Conv2d(128, 128, kernel_size=3, padding=1))

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
            self.res2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(
            ), nn.Conv2d(512, 512, kernel_size=3, padding=1))

            self.classifier = nn.Sequential(nn.MaxPool2d(
                4), nn.Flatten(), nn.Linear(512, num_classes))

        def forward(self, xb):
            out = F.relu(self.conv1(xb))
            out = F.relu(self.conv2(out))
            out = self.res1(out) + out
            out = F.relu(self.conv3(out))
            out = F.relu(self.conv4(out))
            out = self.res2(out) + out
            out = self.classifier(out)
            return out


    # Main loop to query and stream responses
    while True:
        inp = input("User: ")
        if inp == "7697":  # Exit condition
            break
        else:
            stream_llm_answer(
                inp, img="C:\\banana\\codeables\\Projects\\SIH\\image.png")
