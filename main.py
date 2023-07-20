import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
def read_pickle_file(file_path):
    with open(f'{file_path}.pkl', 'rb') as fp:
        return pickle.load(fp)

class facbook_RN_50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self,x):
        return F.sigmoid(self.resnet50(x))

def train(model,train_dataloader,val_dataloader,epochs= 10):
    train_losses = []
    valid_losses = []
    valid_accuracy = []
    #constructing a loop optimizer for training
    optimiser =torch.optim.Adam(model.parameters(), lr=0.01)
    # intialise SummaryWriter
    writer_train = SummaryWriter(log_dir='record/train')  
    writer_val = SummaryWriter(log_dir='record/val') 
    batch_num =0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            features,labels = batch
            features = features.to(device)
            labels = labels.unsqueeze(0)
            labels = labels.to(device)

            predictions = model(features)
            labels= labels.type(torch.LongTensor)
            loss = F.cross_entropy(predictions,labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer_train.add_scalar('loss',loss.cpu().item(),batch_num)
            train_loss += loss.cpu().item() * features.size(0)
            batch_num += 1
    model.eval()
    total_t = 0
    correct_t = 0
    val_loss = 0.0
    with torch.no_grad():
        for val_features, val_labels in val_dataloader:
            val_outputs = model(val_features)
            val_loss = F.cross_entropy(val_outputs, val_labels).item()
            writer_val.add_scalar('loss', val_loss, epoch)
            valid_loss += val_loss * val_features.size(0)
            _,pred_t = torch.max(val_outputs,dim=1)
            correct_t += torch.sum(pred_t == val_labels).item()
            total_t += val_labels.size(0)
        validation_accuracy = 100*correct_t/total_t
        valid_accuracy.append(validation_accuracy)
    train_loss = train_loss/len(train_dataloader.sampler)
    valid_loss = valid_loss/len(val_dataloader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    save_path = os.path.join('./weights_folder', f'epoch_{epoch+1}.pt')
    torch.save(model.state_dict(), save_path)
    metrics = {
            'epoch': epoch + 1,
            'validation_loss': val_loss,
        }
    metrics_path = os.path.join('./model_eval_folder', f'epoch_{epoch+1}_metrics.pt')
    torch.save(metrics, metrics_path)

    writer_train.close() 
    writer_val.close()


csv_path = './data/training_data/training_data.csv'
image_path='./data/training_data/new_images_fb'
dataset = CustomImageDataset(csv_path,img_dir= image_path)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=42)
# Creating dataloaders for all sets of data
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
model = facbook_RN_50()
train(model, train_dataloader, val_dataloader, epochs=10)

