from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from facebook_resnet50 import facbook_RN_50
from torch.nn import Linear
import torch
from torchvision import models
import json 
from tqdm import tqdm
def image_processor(img):
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) 
            ])
    processed_image = transform(img)
    processed_image = processed_image.unsqueeze(0)
    return processed_image
def call_model():
    path = './final_model/model_evaluation/image_model.pt '
    model    = models.resnet50(  weights='IMAGENET1K_V2')
    model.fc = Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load(os.path.join( path)), strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    csv_path = './data/training_data/training_data.csv'
    path = './data/training_data/new_images_fb'
    model = facbook_RN_50()

    df =  pd.read_csv(csv_path)
    lables = df['index'].to_list()
    image_dict= {}
    model =call_model()
    print('pratik')

    for i,j in tqdm(enumerate(lables)):
        im_path = os.path.join(path,str(j))
        img = Image.open(im_path + '.jpg')
        img = image_processor(img)
        emb = model(img)
        image_dict = {i : emb.tolist()[0]}

    json_data = json.dumps(image_dict)    
    with open('image_embeddings.json', 'w') as f:
        f.write(json_data)







