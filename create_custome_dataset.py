import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as transforms
import pickle
from sklearn.model_selection import train_test_split



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.encoder , self.decoder = self.get_encoder_decoder()
        self.img_dir = img_dir
        self.pil_to_tensor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) 
        ])
    def __len__(self):
        return len(self.img_labels)
    def get_encoder_decoder(self):
        category = self.img_labels["category"]
        root_category = category.str.split("/", expand=True)[0].unique()
        encoder = {}
        decoder = {}
        for _ in range(len(root_category)):
            encoder[str.strip(root_category[_])]=_
            decoder[_]= root_category[_]
        return encoder , decoder
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 1]}.jpg')
        image = Image.open(img_path)
        features = self.pil_to_tensor(image)
        label = self.encoder[self.img_labels.iloc[idx, 8]]
        return features, label
def creat_pickle_file(dict,name_dict):
    with open(f'{name_dict}.pkl', 'wb') as fp:
        pickle.dump(dict, fp)
        print('dictionary saved successfully to file')
