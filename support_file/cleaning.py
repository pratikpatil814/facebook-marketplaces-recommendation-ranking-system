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