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


models = ['CM' , 'CM3x3' , 'CN', 'CN3x3' , 'CSD' ,'GLRLM' , 'GLRLM3x3' , 'HOG' , 'LBP', 'LBP3x3']
dim = 256 # number of dimensions 
data = []

#appending all the color model values
for mod in range(0,10):
    ds = pd.read_csv("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Task 1\\{}.csv".format(models[mod]),header = None)
    data.append(ds)
    
col = np.arange(946)    
data_temp = pd.DataFrame(columns = col)

#creating a combined search space of all could model values    
data_temp = data[0]
for i in range(1,len(data)):
    data_temp = pd.concat([data_temp,data[i].iloc[:,1:]], axis = 1)

#Reduce the dimensions to 256
X = data_temp.iloc[:,1:].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA
X_pca = PCA(n_components = dim)
X_PCA = X_pca.fit_transform(X_scaled)

PCA_frame = pd.DataFrame(X_PCA)
data = pd.concat([data_temp.iloc[:,0], PCA_frame], axis = 1)
data.columns = np.arange(data.shape[1])
img_ids = data[0]
img_ids = list(img_ids)

#structure created for cosine similarity comparison
cosine_data = data.T
cosine_data.columns = cosine_data.iloc[0]
cosine_data = cosine_data.iloc[1:,:]


#inputs for the task 5B.
k = int(input("Enter the value for K: "))
#k = 20 # number of bits per signature    
layers = int(input("Enter the value for Layers: "))
#layers = 20 # repeat times
query_id = int(input("Enter the image id: "))
#query_id = 2682134074
t = int(input("Enter the value for t: "))
#t = 10

#signature calculation
random_list = []
layer_list = []
layers_map = []
for run in range(layers):

    randv = numpy.random.randn(dim, k)
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
    
"""
#display dictionaries of all layers.
for i in range(layers):
    print("layer : {}".format(i))
    l = layers_map[i]
    for k,v in l.iteritems():               # will become d.items() in py3k
        print "%s - %s" % (str(k), str(v))
"""     

#signature calculation for the query_image_id
query_map = [] 
query_list = []
for run in range(layers):
    #print(run)
    randv = random_list[run]
    query = np.matmul(cosine_data[query_id].values.tolist(),randv)
    query = (query > 0) * 1
    query_list.append(pd.DataFrame(query))

#creating hash map for each layer
for que in query_list:
    st = str()
    for i in range(len(que.index)):
        st += str(que.loc[i,0])
    query_map.append(st)

#search space for the query_image
search_space = []
for i,l in enumerate(layers_map):
    search_space.append(l[query_map[i]])
final_list = []
for li in search_space:
    final_list += li
#remove duplicates
final_list = set(final_list)

print("The new search space: (Length = {})".format(len(final_list)))
print(final_list)

#similarity calculation
results = []
sim = []

#cosine similarity calculation
for col in (final_list):
    sim.append([col, 1 - cosine(pd.to_numeric(cosine_data[query_id]), pd.to_numeric(cosine_data[col]))])
     
#top t similar images           
results = sorted(sim,  key = lambda x : x[1], reverse=True)[:t]


#displaying top t similar images
tree = ET.parse("C:\\MWDB Project\\devset\\devset_topics.xml")
# get root element 
root = tree.getroot() 
# iterate location items
location_id = dict()
for item in root.findall('./topic'):
    # iterate child elements of item
    for child in item:
        # special checking for name space object content:media
        if child.tag == 'number': 
            num = int(child.text)
        if child.tag == 'title': 
            title = child.text
            location_id[num] = title

img_to_path = dict()
for loc in location_id:
    path ="C:\\MWDB Project\\devset\\img\\img\\{}".format(location_id[loc])
    dirListing = os.listdir(path)
    files=list()
    for file in dirListing:
        if file.replace(".jpg","") not in img_to_path:
            img_to_path[file.replace(".jpg","")] = path + "\\" + file
                        
json.dump(img_to_path, open("C:\\MWDB Project\\devset\\Img_Path.json", 'w'))

img_to_path = json.load(open("C:\\MWDB Project\\devset\\Img_Path.json"))
task_t = "Task 5  t-Similar Images"
f = open('C:\\MWDB Project\\devset\\{}.html'.format(task_t),'w')

message = """<html>
<head><title>{}</title></head>
<body>
<center>
<h1>{}</h1>
</center>
<h3>&nbsp&nbsp&nbsp&nbspGiven Image: </h3>
<br><div style="text-align:center; width:100%">""".format(task_t,task_t)

message = message + """<center><div class="img-with-text"  style="padding: 10px;margin:0 auto;float:left; display: inline-block; height: 300px; width:340px">
                        <img src="{}" height="250" width="350">
                        <p>{}</p>
                        </div></center>""".format(img_to_path[str(query_id)], query_id)

message = message + """
</div>
<div style="text-align:center; clear:left">
<h3 style="text-align:left">&nbsp&nbsp&nbsp&nbspMost Relevant Images Given by LSH: </h3>"""

for i,img in enumerate(results):
        message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340">
                        <img src="{}" height="250" width="350">
                        <p>{}&nbsp&nbsp{}</p>
                        </div>""".format(img_to_path[str(img[0])], img[0],img[1])

message = message + """
</div>
</body>
</html>"""
f.write(message)
f.close()

filename = 'C:\\MWDB Project\\devset\\' + '{}.html'.format(task_t)
webbrowser.open_new_tab(filename)
time.sleep(2)
##################################################################
'''
task = "Task 5 Search Space"
f = open('C:\\MWDB Project\\devset\\{}.html'.format(task),'w')

message = """<html>
<head><title>{}</title></head>
<body>
<center>
<h1>{}</h1>
</center>
<h3>&nbsp&nbsp&nbsp&nbspGiven Image: </h3>
<br><div style="text-align:center; width:100%">""".format(task,task)

message = message + """<center><div class="img-with-text"  style="padding: 10px;margin:0 auto;float:left; display: inline-block; height: 300px; width:340px">
                        <img src="{}" height="250" width="350">
                        <p>{}</p>
                        </div></center>""".format(img_to_path[str(query_id)], query_id)

message = message + """
</div>
<div style="text-align:center; clear:left">
<h3 style="text-align:left">&nbsp&nbsp&nbsp&nbspMost Relevant Images Given by LSH: </h3>"""

for i,img in enumerate(final_list):
        message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340">
                        <img src="{}" height="250" width="350">
                        <p>{}</p>
                        </div>""".format(img_to_path[str(img)], img)

message = message + """
</div>
</body>
</html>"""
f.write(message)
f.close()

filename = 'C:\\MWDB Project\\devset\\' + '{}.html'.format(task)
webbrowser.open_new_tab(filename)
time.sleep(2)

'''