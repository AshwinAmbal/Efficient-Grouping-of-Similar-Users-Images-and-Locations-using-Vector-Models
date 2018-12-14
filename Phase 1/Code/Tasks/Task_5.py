# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 04:38:38 2018

@author: AshwinAmbal

Description: The code below is used to extract 'k' similar locations given a 
location id and a model (CM, CM3x3, CN, CN3x3, CSD, GLRLM) based on visual descriptors. 
For each match, the code also lists the overall matching score and the individual 
contributions of the 10 visual models.
"""

from scipy.spatial.distance import cosine
import csv
from collections import defaultdict
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
file1 = open("C:\\MWDB Project\\Phase 1\\Code\\Sample_Inputs_CMS.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
models = ['CM' , 'CM3x3' , 'CN', 'CN3x3' , 'CSD' ,'GLRLM' , 'GLRLM3x3' , 'HOG' , 'LBP', 'LBP3x3']
for line in list_of_lines:
    words = line.split()
    loc = location_id[int(words[0])]
    k = int(words[1])
    column_sim_model = defaultdict(list)
    for model in models:
        list_of_lines = list()
        csvfile = open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\img_edit_task5\\{}Edit.csv".format(loc + " " + model),"r", encoding="utf8")
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
            csvfile = open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\img_edit_task5\\{}Edit.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
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
            column_sim_sum[c] = [[number, location_id[number]] + [sum_val]]
            c += 1
        for i, column in enumerate(column_sim_sum):
            column_sim_sum[i] = column_sim_sum[i][0]
        sum_temp = sorted(column_sim_sum, key = lambda x : x[2], reverse = True)[:k]
        column_sim_model[model].append(sum_temp)
    
    count_most_occ = dict()
    
    for index in range(1, 31):
        count_most_occ[index] = 0
    
    for key, value in column_sim_model.items():
        for i, req_list in enumerate(value[0]):
            count_most_occ[req_list[0]] += (k-i)
    
    sorted_by_value = sorted(count_most_occ.items(), key=lambda kv: kv[1], reverse = True)[:k]
    
    for item in sorted_by_value:
        print("\n", item[0], location_id[item[0]],"(", item[1], "/", k * 10, ")")
        for model, value in column_sim_model.items():
            for index, tuple_row in enumerate(value[0]):
                if(tuple_row[0] == item[0]):
                    print("Model: ", model, ", Similarity Score: ", tuple_row[2], ", Rank: ", index + 1)
