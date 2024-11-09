#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


__locations = None

__data_columns = None

__model = None

def get_estimated_price(location,sqft,bath,bhk):

    df5=pd.read_csv('data.csv')
    if location not in __locations:
        location = 'other'
    new_row = {'location': location, 'total_sqft': sqft, 'bath': bath, 'bhk': bhk}
    input_df = pd.concat([df5, pd.DataFrame([new_row])], ignore_index=True)
    le = LabelEncoder()
    input_df.location=le.fit_transform(input_df.location)
    input_df=input_df.values
    preprocessor = ColumnTransformer(transformers=[('location', OneHotEncoder(sparse=False), [0])],remainder='passthrough')
    input_df_pre = preprocessor.fit_transform(input_df)
    scaler1 = StandardScaler(with_mean=False)
    new_scaled = scaler1.fit_transform(input_df_pre)
    last_row_transformed = new_scaled[-1].reshape(1, -1)

    return __model.predict(last_row_transformed)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[:-3]
    global __model
    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())

    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))

    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))

    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location

    print(get_estimated_price('Ejipura', 1000, 2, 2)) # other location


# In[ ]:




