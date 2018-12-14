# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 02:15:26 2018

@author: AshwinAmbal
Description: The code below is used to extract 'k' similar images for a given image 
based on the tf, df or idf values as specified in the sample inputs file.
"""

import pandas as pd
from scipy.spatial.distance import cosine
#from scipy.spatial.distance import euclidean
#import pickle

#final_annot_tf = list()
#final_annot_df = list()
#final_annot_idf = list()

# Loading pickle file for faster execution

#pickle.load(final_annot_tf , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_tf.p", "rb" ))
#pickle.load(final_annot_df , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_df.p", "rb" ) )
#pickle.load(final_annot_idf , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_idf.p", "rb" ) )

# Reading the sample input file

file1 = open("C:\\MWDB Project\\Phase 1\\Code\\Sample_Inputs_Img.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
for line in list_of_lines:
    words = line.split()
    img = words[0]
    input_type = words[1]
    input_type = input_type.replace("-", "_")
    k = int(words[2])
    # Reading the csv file into the dataframe
    df = pd.read_csv("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\{}_Img.csv".format(input_type))
    
    # Finding cosine similarity between the columns
    i = 0
    sim = list()
    for column in df:
        if column != "Annotations" and column != img:
            sim.append([1 - cosine(df[img], df[column]), column])
    
    results = sorted(sim, reverse=True)[:k]
    
    final_list = list()
    for row in results:
        temp_list = list()
        for annot, data1, data2 in zip(df["Annotations"], df[img], df[row[1]]):
            if data1 != 0 and data2 != 0:
                temp_list.append([annot, abs(data1 - data2)])
        temp_list = sorted(temp_list, key = lambda x : x[1])[:3]
        final_list.append(temp_list)
    
    print("\n\nMost Similar Images for Image ID = ", img,", k = ", k, ", Metric = ", input_type)
    print("\nUSING COSINE SIMILARITY:")
    for j in range(0,k):
        print(results[j] + final_list[j])
    
    print()
"""
    # Finding euclidean similarity between the columns
    sim_euc = list()
    for column in df:
        if column != "Annotations" and column != img:
            sim_euc.append([1 - euclidean(df[column], df[img]), column])
            
    results_euc = sorted(sim_euc, reverse = True)[:k]
    
    final_list = list()
    for row in results_euc:
        temp_list = list()
        for annot, data1, data2 in zip(df["Annotations"], df[img], df[row[1]]):
            if data1 != 0 and data2 != 0:
                temp_list.append([annot, abs(data1 - data2)])
        temp_list = sorted(temp_list, key = lambda x : x[1])[:3]
        final_list.append(temp_list)
    
    print("USING EUCLIDEAN SIMILARITY:")     
    for j in range(0,k):
        print(results_euc[j] + final_list[j])
"""