from io import BytesIO

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from helper_lib.model import get_model


app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = get_model("CNN").to(device)

checkpoint_path = "checkpoints/best/model_epoch_003.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.get("/")
def read_root():
    return {
        "message": "FastAPI text generation, embedding, and image classification API is running"
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    result = embedding_model.get_embedding(request.word)
    return result


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted_index.item()]

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4)
    }