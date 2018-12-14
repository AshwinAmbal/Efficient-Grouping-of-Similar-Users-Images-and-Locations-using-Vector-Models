# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 02:15:26 2018

@author: AshwinAmbal

Description:The code below is used to extract 'k' similar location for a given 
location based on the tf, df or idf values as specified in the sample inputs file.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from scipy.spatial.distance import cosine
#from scipy.spatial.distance import euclidean
#from sklearn.metrics.pairwise import cosine_distances
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
            
# Reading the sample input file
file1 = open("C:\\MWDB Project\\Phase 1\\Code\\Sample_Inputs_Loc.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
for line in list_of_lines:
    words = line.split()
    loc = location_id[int(words[0])]
    input_type = words[1]
    input_type = input_type.replace("-", "_")
    k = int(words[2])
    # Reading the csv file into the dataframe
    df = pd.read_csv("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\{}_Loc.csv".format(input_type))
    
    # Finding cosine similarity between the columns
    i = 0
    sim = list()
    for column in df:
        if column != "Annotations" and column != loc:
            sim.append([1 - cosine(df[loc], df[column]), column])
    
    results = sorted(sim, reverse=True)[:k]
    
    final_list = list()
    for row in results:
        temp_list = list()
        for annot, data1, data2 in zip(df["Annotations"], df[loc], df[row[1]]):
            if data1 != 0 and data2 != 0:
                temp_list.append([annot, abs(data1 - data2)])
        temp_list = sorted(temp_list, key = lambda x : x[1])[:3]
        final_list.append(temp_list)
    
    print("\n\nMost Similar Locations for Loc ID = ", words[0], ", Location = ", loc, ", k = ", k, " Metric = ", input_type)
    print("\nUSING COSINE SIMILARITY:")
        
    for j in range(0,k):
        print(results[j] + final_list[j])
    
    print()
"""    
    # Finding euclidean similarity between the columns
    sim_euc = list()
    for column in df:
        if column != "Annotations" and column != loc:
            sim_euc.append([1 - euclidean(df[column], df[loc]), column])
            
    results_euc = sorted(sim_euc, reverse = True)[:k]
    
    final_list = list()
    for row in results_euc:
        temp_list = list()
        for annot, data1, data2 in zip(df["Annotations"], df[loc], df[row[1]]):
            if data1 != 0 and data2 != 0:
                temp_list.append([annot, abs(data1 - data2)])
        temp_list = sorted(temp_list, key = lambda x : x[1])[:3]
        final_list.append(temp_list)
        
    print("USING EUCLIDEAN SIMILARITY:")
    for j in range(0,k):
        print(results_euc[j] + final_list[j])
""" 