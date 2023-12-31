# Facebook Marketplace AI System

This project is an implementation of an AI system behind Facebook Marketplace, a platform for buying and selling products on Facebook. The system utilizes artificial intelligence techniques to provide personalized recommendations for relevant listings based on user search queries. 

## Introduction
Facebook Marketplace AI System is designed to enhance the user experience by leveraging AI algorithms to recommend the most relevant product listings. By understanding user preferences and search queries, the system aims to assist users in finding the products they are looking for efficiently and effectively.

## Features
- Personalized Recommendations: The AI system utilizes user data and preferences to provide personalized recommendations tailored to individual users.
- Intelligent Search: The system employs advanced search algorithms to optimize the search process and deliver accurate results.
- Seamless Integration: The codebase can be easily integrated with the existing Facebook Marketplace infrastructure for a seamless user experience.
- Scalability: The system is designed to handle a large number of users and listings efficiently, ensuring scalability and performance.

## Usage
Once the Facebook Marketplace AI System is up and running, users can access it through the Facebook Marketplace interface. The system will automatically analyze their search queries and provide personalized recommendations based on their preferences.

# Data Cleaning - Tabular Dataset

In this section, we will perform data cleaning on the tabular dataset provided in the AWS EC2 instance. Follow the steps below to clean the dataset and prepare it for further analysis.

## SSH into the EC2 Instance
To access the dataset and perform data cleaning, follow these steps:

1. Open a terminal or command prompt on your local machine.

2. Change the permissions of the private key file:
   ```shell
   chmod 400 path/to/private_key.pem
   ```

3. SSH into the EC2 instance using the private key:
   ```shell
   ssh -i path/to/private_key.pem ec2-user@<EC2-Instance-Public-IP>
   ```

   Replace `<EC2-Instance-Public-IP>` with the public IP address of your EC2 instance. Note that you may need to use a different username (`ubuntu`, `centos`, etc.) depending on the EC2 instance configuration.

4. You should now be connected to the EC2 instance via SSH.

## Cleaning the Tabular Dataset
Now that you are connected to the EC2 instance, let's clean the tabular dataset using the provided `clean_tabular_data.py` file.


```python
#data_extraction code
import pandas as pd
class dataextractor:
    def data_extract_csv(self,path):
        df = pd.read_csv(path,lineterminator='\n')
        return df
# data_cleaning code
import pandas as pd

class datacleaning:
    
    def clean_csv_file(self,df):
        df = df.drop(df.columns[0],axis= 1)
        df.dropna(subset='price',how='any',inplace=True)
        df.reset_index(inplace=True)
        df['price'] = df['price'].str.replace('£','')
        df['price'] = df['price'].str.replace(',','')
        df['price'] = pd.to_numeric(df['price'])
        return df
# clean_tabular_data.py
import pandas as pd
import numpy as np
from support_file.extraction import dataextractor as de
from support_file.cleaning import datacleaning as dc
path_product = './data/training_data/Products.csv'
path_image = './data/training_data/Images.csv'

df = de().data_extract_csv(path_product)
df = dc().clean_csv_file(df)

# Sanbox.py for to create ecoder and decode
df_image = de().data_extract_csv(path_image)
new_test_data_csv = pd.merge(df,df_image,left_on = 'id',right_on='product_id',how = 'inner')
#new_test_data_csv.info()
new_test_data_csv.rename(columns={'id_x':'label'},inplace='True')
new_test_data_csv.info()
new_test_data_csv.to_csv('./data/training_data/training_data.csv')
category = df["category"]
root_category = category.str.split("/", expand=True)[0].unique()
cat_encoder = {}
cat_decoder = {}
for _ in range(len(root_category)):
    cat_encoder[root_category[_]]=_
    cat_decoder[_]= root_category[_]
print(cat_decoder)
   ```

# Image Dataset Cleaning

In this section, we will perform cleaning operations on the image dataset. The objective is to ensure that all images have the same size and number of channels, achieving consistency across the dataset.

## Prerequisites
Before proceeding, make sure you have completed the following steps:
- Accessed the AWS EC2 instance using SSH and downloaded the dataset, including the folder containing the images.

## Cleaning the Image Dataset
To clean the image dataset, we will create a Python script named `clean_images.py` within your repository. Follow the steps below to perform the necessary cleaning operations.

1. Create a file named `clean_images.py` within your repository.

2. Open `clean_images.py` using a text editor and add the following code:

   ```python
   import os
   from PIL import Image

   def resize_image(final_size, im):
       size = im.size
       ratio = float(final_size) / max(size)
       new_image_size = tuple([int(x * ratio) for x in size])
       im = im.resize(new_image_size, Image.LANCZOS)
       new_im = Image.new("RGB", (final_size, final_size))
       new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
       return new_im

   def clean_image_data(folder_path, final_size):
       # Create a new folder for cleaned images
       cleaned_folder_path = os.path.join(folder_path, "cleaned_images")
       os.makedirs(cleaned_folder_path, exist_ok=True)

       # Get the list of image files in the folder
       image_files = os.listdir(folder_path)

       for n, item in enumerate(image_files, 1):
           image_path = os.path.join(folder_path, item)
           im = Image.open(image_path)
           new_im = resize_image(final_size, im)
           new_image_path = os.path.join(cleaned_folder_path, f"{n}_resized.jpg")
           new_im.save(new_image_path)
   ```

   This code defines a `resize_image` function that resizes the input image to a specified `final_size` while maintaining the aspect ratio. It then pads the resized image to make it square. The `clean_image_data` function takes a `folder_path` parameter as the path to the folder containing the images. It creates a new folder named "cleaned_images" to store the cleaned images. The function loops through each image, resizes it using the `resize_image` function, and saves the cleaned image to the new folder with a modified filename.

3. Save the `clean_images.py` file.

## Model Construction and Transfer Learning
The purpose of this project is to provide feedback in the form of recommendations related to a specific product. In order to accomplish this goal, the first step will be to develop an image categorization model. Since we are working with photos, it would be beneficial to employ a deep learning model in this situation. I did this by utilising a convolutional neural network; however, rather than developing my very own network from scratch, I have used the most of models that were already available.

```python
class facbook_RN_50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self,x):
        return F.sigmoid(self.resnet50(x))
        
```

## Developing the training loop
Since we have now developed both our model and our dataset, we are in a position to make use of a training loop in order to arrive at the weights of our model. These weights will be put to use during the stage in which we extract features. In order to accomplish this, we will iteratively proceed through our train dataloader; for each batch, we will generate a prediction, which will subsequently be utilised in the computation of a loss. After then, this information is relayed backwards through the neural network. A training loss and a validation loss, as well as an accuracy measure for the validation, are computed for each epoch. This, together with the visualisations we create with our tensorboards, is utilised to determine which epoch should be used for the model weights. 

``` python


def train_model(model,data_loaders,criterion,optimizer,epochs=2):
        # Create a folder for the model evaluation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_eval_folder = f'model_evaluation_loaded_weight/{timestamp}'
    os.makedirs(model_eval_folder, exist_ok=True)

    # Create a folder within model evaluation for the model weights of each epoch
    weights_folder = os.path.join(model_eval_folder, 'weights')
    os.makedirs(weights_folder, exist_ok=True)

    writer_train = SummaryWriter(log_dir='record/train')  
    writer_val = SummaryWriter(log_dir='record/val') 
    for epoch in range(epochs):
        print('Epioch %d / %d'%( epoch,epoch-1))
        print('-'*15)
        device = T.device("cuda" if T.cuda.is_available() else "cpu")

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct = 0
            model = model.to(device)
            pbar = tqdm(data_loaders[phase])

            for inputs , labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                optimizer.zero_grad()
                with T.set_grad_enabled(phase == 'train'):
                    ouputs = model(inputs)
                    loss = F.cross_entropy(ouputs,labels)

                    _,preds = T.max(ouputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                correct += T.sum(preds == labels.data)
                if phase == 'train':
                    writer_train.add_scalar('loss',loss.cpu().item(),epoch)
                else:
                    writer_val.add_scalar('loss', running_loss, epoch)
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = correct.double()/len(data_loaders[phase].dataset)

            print('{} Loss: {:.4f} acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            save_path = os.path.join(model_eval_folder, f'epoch_{epoch+1}.pt')
            T.save(model.state_dict(), save_path)
            metrics = {
                    'epoch': epoch + 1,
                    'validation_loss': running_loss,
                }
            metrics_path = os.path.join(model_eval_folder, f'epoch_{epoch+1}_metrics.pt')
            T.save(metrics, metrics_path)

            writer_train.close() 
            writer_val.close()

if __name__ == '__main__':
    csv_path = './data/training_data/training_data.csv'
    image_path='./data/training_data/new_images_fb'
    dataset = CustomImageDataset(csv_path,img_dir= image_path)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=42)
    train_loader = T.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    val_loader = T.utils.data.DataLoader(val_data, batch_size=2, shuffle=True, num_workers=4)
    data_loaders = {'train': train_loader, 'val': val_loader}

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    model = facbook_RN_50()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr =0.001)
    weights_path = './weights_folder/epoch_1.pt'
    model.load_state_dict(T.load(weights_path))
    train_model(model,data_loaders,criterion,optimizer)
    print(data_loaders)


```

## High-Level Features Extraction
Our previous work built an image classification model, however this is not what we need for our eventual aim of comparing similar products. In actuality, we need to extract high-level features from each image, i.e. we need a vector for each image. We do picture classification to begin with in order to acquire better vector embeddings for each image (remember, the resnet50 is not finetuned to our goal). We only get this if we delete the last two completely linked layers of the neural network.

``` python
from facebook_resnet50 import facbook_RN_50
import torch
import os

model = facbook_RN_50()
weights_path = './weights_folder/epoch_1.pt'
model.load_state_dict(torch.load(weights_path))
model = torch.nn.Sequential(*list(model.children())[:-2])

save_path = 'final_model/model_evaluation/image_model.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
```
## Retrieving Image Embeddings
The image_processor.py script preprocesses single images for machine learning model input. (batch_size, n_channels, height, width) photos were utilised to train the model. When inferring with a single image, the input shape must be (n_channels, height, width).This script seamlessly transforms the input image using the same preprocessing procedures as training. It formats the image for the model's input. The script resizes, normalises, and transforms images using PIL and torchvision.transforms.To utilise the script, enter the image path in the image_path variable. The script preprocesses the image and adds a batch dimension to make a batch of size 1. Your machine learning model can immediately infer from this preprocessed image.Image_processor.py lets you easily integrate single-image inputs into your model pipeline without preprocessing, assuring accurate and efficient predictions.

``` python
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
```

In the previous phase, we started by saving the image_id's and corresponding embeddings into a pickle file. Afterward, we loaded this pickle file into memory as a variable. Next, we separated the image_id's and embeddings from the loaded variable, creating two distinct arrays. However, we encountered an issue with the arrays' shapes not being consistent. To address this problem, we introduced additional code to ensure that the arrays were made homogeneous, having consistent shapes throughout. By doing so, we successfully resolved the shape-related discrepancies in the data, enabling smoother processing and analysis.

``` python
import faiss               
import json
import numpy as np
class faiss_class():
    def get_dic_to_numpy_array(self):
        with open('image_embeddings.json') as json_file:
            dic = json.load(json_file)
        matrix = np.empty([0,13])
        for key in dic.keys():
            matrix = np.vstack([matrix,np.array(dic[key], dtype=np.float32)])
        return matrix

    def get_FAISS_index(self):
        index = faiss.IndexFlatL2(13) 
        index.add(self.get_dic_to_numpy_array())   
        return index

if __name__ == '__main__':
    fa = faiss_class()
    index = fa.get_FAISS_index()
    faiss.write_index(index,'FAISS.pkl')

```