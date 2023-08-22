import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
from  torchvision import models
import os
import faiss
import image_processor

class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(  weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 13)
        self.model.eval()
    
        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str



try:
    feature_extraction = FeatureExtractor()
    feature_extraction.model.load_state_dict(torch.load(os.path.join('final_model','image_model.pt' ),map_location=torch.device('cpu')), strict=False)
    feature_extraction.model.eval()
    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    index = faiss.read_index("FAISS.pkl")
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    tem_image = image_processor(pil_image)
    img_embeddded = feature_extraction(tem_image)

    return JSONResponse(content={
    "features": img_embeddded.tolist()[0], # Return the image embeddings here
        })
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    tem_image = image_processor(pil_image)
    img_embeddded = feature_extraction(tem_image)
    _, I = index.search(img_embeddded.detach().numpy(), 4) 
    key = list(img_embeddded.keys())
    img_labels = []
    for _ in I.tolist()[0]:
        img_labels.append(key[_])



    return JSONResponse(content={
    "similar_index":  I.tolist()[0],
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)