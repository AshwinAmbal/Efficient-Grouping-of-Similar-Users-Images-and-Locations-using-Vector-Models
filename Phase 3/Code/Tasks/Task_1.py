"""
    The code below forms a image-image similarity graph for the images given in the dataset such that
    for each image there are 'k' outgoing edges to k-different images which are most similar to the
    image at hand. The similarity measure used is cosine similarity.
    Caution: Code can take extremely long time to run because of running on large dataset and 
    combining scores from all the models.
"""

import csv
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.spatial.distance import cosine
from collections import defaultdict
from datetime import datetime
import json

# Getting all the location ID's and their corresponding names in a dictionary
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
            

# Combining all the imagees from different locations into a single file for each color model separately.
file_numbers = list(range(1, len(location_id) + 1))  
models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
for model in models:
    final_list_of_lines = list()
    img_seen_so_far = list()
    for number in file_numbers:
        csvfile = open("C:\\MWDB Project\\devset\\descvis\\img\\{}.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
        reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
        list_of_lines = list(reader)
        for line in list_of_lines:
            if line[0] not in img_seen_so_far:
                final_list_of_lines.append(line)
            img_seen_so_far.append(line[0])
    with open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Task 1\\{}.csv".format(model), 'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter=',')
        for img in final_list_of_lines:
            writer.writerow(img)


models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']

column_sim_model = defaultdict(list)

k = int(input("Enter the value  of k: "))

# Forming the image-image similarity graph by combing similarity scores given by each model for every image in dataset
print("Started at: ", datetime.now().time().strftime('%H:%M:%S'))
for model in models:
    csvfile = open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Task 1\\{}.csv".format(model),"r", encoding="utf8")
    reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
    list_of_lines = list(reader)
    loader_list = list(map(list, zip(*list_of_lines)))
    
    df = pd.DataFrame(loader_list)
    df.columns = df.iloc[0]
    df = df[1:]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column])
    
    sim_dict = dict()
    for i, column1 in enumerate(df.columns):
        comp_df = list(df.columns[i+1 : ])
        sim = list()
        for column2 in comp_df:
            similarity = 1 - cosine(pd.to_numeric(df[column1]), pd.to_numeric(df[column2]))
            sim.append([column1, column2, similarity])
            sim_dict[(column1, column2)] = similarity
        comp_df.append(column1)
        prev_set_of_col = list(set(df.columns) - set(comp_df))
        for column in prev_set_of_col:
            sim.append([column1, column, sim_dict[(column, column1)]])
        results = sorted(sim, key = lambda x : x[2], reverse=True)[:k]        
        if column_sim_model[column1] != []:
            for result in results:
                flag = 0
                for r, row in enumerate(column_sim_model[column1]):
                    if row[0] == result[1]:
                        column_sim_model[column1][r][1] += result[2]
                        flag = 1
                        break
                if flag == 0:
                    column_sim_model[column1].append(result[1:])
        else:
            column_sim_model[column1] = [res[1:] for res in results]
        print("Finished with : ", i, " " , column1, " at : ",datetime.now().time().strftime('%H:%M:%S'))
    print()
    print("Done with : ", model, " at: ", datetime.now().time().strftime('%H:%M:%S'))

# Default dictionary containing image-image similarity graph with keys as nodes and values as list of
# list of adjacent nodes, similarity scores
# Here similarity scores give the edge weights
img_sim_graph = column_sim_model.copy()
for key, value in column_sim_model.items():
    column_sim_model[key] = sorted(value, key = lambda x : x[1], reverse = True)[:k]
    
# writing
json.dump(column_sim_model, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(k), 'w'))

# reading
#column_sim_model = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(k)))