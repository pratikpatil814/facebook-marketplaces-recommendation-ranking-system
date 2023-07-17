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


