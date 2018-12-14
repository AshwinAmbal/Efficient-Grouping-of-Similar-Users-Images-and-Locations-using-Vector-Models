import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

k = input("Enter the value of K")
column_sim_model = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(k)))                            
                             
count = 0
for i in column_sim_model.keys():
    count=count+1
a = np.zeros(shape=(count,count))
imageids=[]
for x in column_sim_model.keys():
    imageids.append(x)
image_dict = dict()
listt = [[0] * len(imageids) for _ in range(1,len(imageids)+1)]
for i,img in enumerate(imageids):
    image_dict[img]=i
for key,value in column_sim_model.items():
    for val in value:
        listt[image_dict[key]][image_dict[val[0]]] = val[1]
df=pd.DataFrame(listt)
df.columns=imageids
df.index=imageids
                             
#choice = input("Enter Option.  1)Nearest Neighbour  2)Spectral Clustering ")                            
choice = 1
if(choice=='1'):
    centroids=[]
    c=int(input("Enter number of clusters"))
    # Replace df with tempdf while trying for smaller samples
    for i in range(c):
        randomsamplenumber = random.randint(0,len(df))
        centroids.append(df.iloc[randomsamplenumber,:].tolist())

    distancesdict=dict()
    for i in range(c):
        distancesxx=[]
        for j in range(len(df.index)):
            dist = distance.euclidean(centroids[i],df.iloc[j,:].tolist())
            distancesxx.append(dist)
        distancesdict['Centroid'+str(i)] = distancesxx    

    means=[]
    for i in range(c):
        men = np.average(distancesdict['Centroid'+str(i)])
        means.append(men)

    tempdict=dict()
    for cluster,mean in zip(centroids,means):
        tempdict[mean]=cluster
    temp=sorted(tempdict)
    means=list()
    centroids=list()
    for value in temp:
        centroids.append(tempdict[value])
        means.append(value)

    clusters= dict()
    # Replace df with tempdf while trying for smaller samples
    dfcopy = df
    for i in range(len(centroids)):
        df = dfcopy
        cluster=[]
        for j, img in enumerate(df.index):
            dist = distance.euclidean(centroids[i],df.loc[img,:].tolist())
            if((dist)<= means[i]):
                cluster.append([img, dist])
                dfcopy = dfcopy.drop(index=img)
        clusters['Cluster' + str(i)] = cluster
#     Visualizing
    import xml.etree.ElementTree as ET
    import os
    import json
    import webbrowser
    import time

    tree = ET.parse("C:\\MWDB Project\\devset\\devset_topics.xml")
    # get root element 
    root = tree.getroot() 
    # iterate location items
    location_id = dict()
    for item in root.findall('./topic'):
        # iterate child elements of item
        for child in item:
            # special checking for namespace object content:media
            if child.tag == 'number': 
                num = int(child.text)
            if child.tag == 'title': 
                title = child.text
                location_id[num] = title

    img_to_path = dict()        
    for loc in location_id:
        path ="C:\\MWDB Project\\devset\\img\\{}".format(location_id[loc])
        dirListing = os.listdir(path)
        files=list()
        for file in dirListing:
            if file.replace(".jpg","") not in img_to_path:
                img_to_path[file.replace(".jpg","")] = path + "\\" + file

    json.dump(img_to_path, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json", 'w'))

    img_to_path = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json"))

    for k in range(c):
        images=[]        
        for i in clusters['Cluster'+str(k)]:
            images.append(i[0])

        # images = ['3103006230','2718956990','12497387424','1675682410','1832536520', '2907638727', '1832538318', '223803485']
        task = "Task 2 {}".format(k)
        f = open('C:\\MWDB Project\\Phase 3\\Code\\CSV\\{}.html'.format(task),'w')

        message = """<html>
        <head><title>{}</title></head>
        <body>
        <center>
        <h1>{}</h1>
        <h2>Cluster {}</h2>
        </center>
        <br><div style="text-align:center">""".format(task,task,k)

        for i,img in enumerate(images):
                message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340">
                                <img src="{}" height="250" width="350">
                                <p>{}</p>
                                </div>""".format(img_to_path[img], img)

        message = message + """
        </div>
        </body>
        </html>"""

        f.write(message)
        f.close()

    #     filename = 'D://MWD Project//Phase3//Code//CSV//' + 

        filename = 'C://MWDB Project//Phase 3//Code//CSV//' + '{}.html'.format(task)
        webbrowser.open_new_tab(filename)
        time.sleep(2)
"""
if(choice=='2'):
    scaled_data = preprocessing.scale(df.T)
    pca = PCA(n_components = 1000)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    pca.n_components_
    print()
    print("\nTotal Variance Accounted for: ", (pca.explained_variance_ratio_* 100).sum())
    new_df = pd.DataFrame(pca_data)
    new_df = new_df.T
    new_df.columns = df.columns
    clustering = SpectralClustering(n_clusters= c,assign_labels="discretize",random_state=0).fit(new_df)
    X = clustering.labels_
    for i in range(0,1000):
        print("The image" ,i, " belongs to the cluster: " , X[i])
    # for index, row in df.iterrows():
    #     for i in range(8912):
    #         if(row[i] > 0):
    #             row[i] = 1

    # import networkx as nx
    # G = nx.DiGraph(df)
    # L = nx.directed_laplacian_matrix(G)
    # print(L)
    # L.size
    # M = np.array(L)
    # from numpy import linalg
    # e,v=np.linalg.eig(L)
    # eigenvalue = v
    # idy = v.argsort()[100][::-1]
    # m = m[idy]
    # n = n[:,idy]
"""