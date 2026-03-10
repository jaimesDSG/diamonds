
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_REGISTRY 

import pickle 
import os 

def save_model(model, path):
    if not os.path.exists(path) : 
        os.mkdir(path)
    with open(os.path.join(path,"model.pkl"),"wb")  as f:
        pickle.dump(model,f)

def load_model(path) -> BaseEstimator:
    with open(os.path.join(path,"model.pkl"),"rb")  as f:
        return pickle.load(f)