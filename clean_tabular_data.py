from support_file.extraction import dataextractor as de
from support_file.cleaning import datacleaning as dc
path_product = './data/training_data/Products.csv'
path_image = './data/training_data/Images.csv'

df = de().data_extract_csv(path_product)
df = dc().clean_csv_file(df)
df_image = de().data_extract_csv(path_image)
