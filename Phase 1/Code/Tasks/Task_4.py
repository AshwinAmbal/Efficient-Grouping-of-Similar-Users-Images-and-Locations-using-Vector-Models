# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 04:38:38 2018

@author: AshwinAmbal

Description: The code below is used to extract 'k' similar locations given a 
location id and a model (CM, CM3x3, CN, CN3x3, CSD, GLRLM, etc). For each match, 
the code also lists the overall matching score as well as the 3 image pairs that 
have the highest similarity contribution.
"""

from scipy.spatial.distance import cosine
import csv
import xml.etree.ElementTree as ET
import pandas as pd
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

# Reading the sample input file and computing similarity for each input   
file1 = open("C:\\MWDB Project\\Phase 1\\Code\\Sample_Inputs_CM.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
for line in list_of_lines:
    words = line.split()
    loc = location_id[int(words[0])]
    input_type = words[1]
    k = int(words[2])
    
    list_of_lines = list()
    
    csvfile = open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\img_edit\\{}Edit.csv".format(loc + " " + input_type),"r", encoding="utf8")
    reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
    list_of_lines = list(reader)
    
    loader_list = list(map(list, zip(*list_of_lines)))
    
    df = pd.DataFrame(loader_list)
    df.columns = df.iloc[0]
    file_numbers = list(range(1, 31))
    file_numbers.remove(int(words[0]))
    c = 0
    column_sim_sum = [0] * 29
    for number in file_numbers:
        list_of_lines = list()
        csvfile = open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\img_edit\\{}Edit.csv".format(location_id[number] + " " + input_type),"r", encoding="utf8")
        reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
        list_of_lines = list(reader)
        loader_list = list(map(list, zip(*list_of_lines)))
        comp_df = pd.DataFrame(loader_list)
        comp_df.columns = comp_df.iloc[0]
        result = list()
        for column1 in df.columns:
            sim = list()
            temp_sim = list()
            for column2 in comp_df.columns:
                temp_sim = [column1, column2, 1 - cosine(pd.to_numeric(df[column1][1:]), pd.to_numeric(comp_df[column2][1:]))]
                sim.append(temp_sim)
            result.append(sorted(sim, key = lambda x : x[2], reverse = True)[0])
            comp_df = comp_df.drop(sorted(sim, key = lambda x : x[2], reverse = True)[0][1], 1)
        sum_val = 0
        for row in result:
            sum_val += row[2]
        column_sim_sum[c] = [[number, location_id[number]] + [sum_val] + [sorted(result, key = lambda x : x[2], reverse = True)[:3]]]
        c += 1
    for i, column in enumerate(column_sim_sum):
        column_sim_sum[i] = column_sim_sum[i][0]
    
    sum_temp = sorted(column_sim_sum, key = lambda x : x[2], reverse = True)[:k]
    print("\n\n Similarity score for location id: ", words[0], " location name: ", loc, " Model: ", input_type, " k: ", k)
    print("\nCosine Similarity: ")
    for row in sum_temp:
        print(row)
        print()