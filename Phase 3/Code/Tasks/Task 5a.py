import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import numpy
import json
from scipy.spatial.distance import cosine
import webbrowser
import os
import time



#inputs for the task 5A.
k = int(input("Enter the value for K: "))
#k = 20 # number of bits per signature    
layers = int(input("Enter the value for Layers: "))
#layers = 20 # repeat times
random_list = []
layer_list = []
layers_map = []
num = int(input("Enter the number of input vectors: "))
num_dim = int(input("Enter the number of dimensions of the input vector: "))

#creating the input vector space
vector_list = []
for i in range(num):
    ids = int(input("Enter id: "))
    vector = [ids]
    for j in range(num_dim):
        val = float(input("value = "))
        vector.append(val)
    vector_list.append(vector)
data = pd.DataFrame(vector_list)
img_ids = data[0]
img_ids = list(img_ids)

#signature calculation for the query_image_id
for run in range(layers):
    randv = numpy.random.randn(num_dim, k)
    random_list.append(randv)
    layer = np.matmul(data.iloc[:,1:].values.tolist(),randv)
    layer = (layer > 0) * 1
    layer_list.append(pd.DataFrame(layer))
 
#creating hash map for each layer
layers_map = []
for lay in layer_list:
    layer_map = defaultdict(list) 
    for ind in lay.index:
        st = str()
        for col in lay.columns:
            st += str(lay.loc[ind,col])
        layer_map[st].append(img_ids[ind])
    layers_map.append(layer_map)

for i in range(layers):
    print("\n")
    print("layer : {}".format(i))
    l = layers_map[i]
    for k,v in l.items():               # will become d.items() in py3k
        print (str(k), str(v))
        