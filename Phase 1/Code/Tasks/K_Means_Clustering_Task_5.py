# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 04:38:38 2018

@author: AshwinAmbal

    Description: The code below performs K-Means Clustering with a cluster size of 7
    with 10 image samples taken from each cluster and written into the edited images 
    folder. These images are used for Task_5.
"""

#importing libaries
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from collections import defaultdict
import csv

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

models = ['CM' , 'CM3x3' , 'CN', 'CN3x3' , 'CSD' ,'GLRLM' , 'GLRLM3x3' , 'HOG' , 'LBP', 'LBP3x3']
number_clusters = 7
for loca in range(1,31):
    for mod in range(0,10):
        #importing dataset
        ds = pd.read_csv("C:\\MWDB Project\\devset\\descvis\\img\\{}.csv".format(location_id[loca] + " " + models[mod]))
        df = ds.iloc[:,:].values
        X = ds.iloc[:,1:].values

        #k-means clustering
        kmeans = KMeans(n_clusters=number_clusters)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        t_means = y_kmeans
        means = defaultdict(list)
        for loc, clust in enumerate(t_means):
            means[clust].append(loc)
        y_kmeans
        temp = list()
        final_list = list()
        for i in range(0,1000):
            if len(final_list) == number_clusters * 10 :
                break
            if len(means[i%number_clusters]) != 0:
                final_list.append(list(df[means[i%number_clusters].pop()]))
            
        with open('C:\\MWDB Project\\Phase 1\\Code\\CSV\\img_edit_task5\\{}.csv'.format(location_id[loca] + " " + models[mod] + 'Edit'), 'w', encoding = 'utf8', newline='') as outcsv:   
            #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',',quotechar = '"')
            for item in final_list:
                #Write item to outcsv
                writer.writerow(item)