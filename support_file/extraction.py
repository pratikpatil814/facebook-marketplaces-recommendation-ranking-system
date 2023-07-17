import pandas as pd
class dataextractor:
    def data_extract_csv(self,path):
        df = pd.read_csv(path,lineterminator='\n')
        return df