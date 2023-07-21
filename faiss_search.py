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