# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:42:06 2018

@author: AshwinAmbal

Description: The code below is used to extract required data from the file 
devset_textTermsPerImage.txt and write the required information in a database type
format into a csv file. The column header in the written file would signify the
user id's of the various users and the row headers would be annotations.
Eg:
    [Annotations ImageID1 ImageID2 ImageID3]
    ["wicked"         ID1      ID2      ID3]
    [......          ....     ....      ...]
"""



import csv
import pickle

# Opening file to read from
file1 = open("C:\\MWDB Project\\devset\\desctxt\\devset_textTermsPerImage.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
#Reading all image id's into 'ids'
ids = list()
for row in list_of_lines:
    ids.append(row.split()[0])

# Reading all annotations into 'annotation'
annotation = list()
for row in list_of_lines:
    line = row.split()
    for i in range(1, len(line), 4):
        line[i] = line[i].replace("\'", " ")
        line[i] = line[i].replace('"', '')
        line[i] = line[i].replace(',', ' ')
        annotation.append(line[i])

annotation = list(set(annotation))

# Making a dictionary of annotation mapping to the index in which it occurs in the
# list 'annotation'
dict_annot = dict()
for i, annot in enumerate(annotation):
    dict_annot[annot] = i

# Reading line by line from the file and writing the ImageID and their corresponding
# tf, df and tf-idf values as rows and finally taking a transpose of the matrix
# formed to get the required database schema mentioned above.
annot_values_tf = [["Annotations"] + annotation]
annot_values_df = [["Annotations"] + annotation]
annot_values_idf = [["Annotations"] + annotation]

for row in list_of_lines:
    line = row.split()
    flag = 0
    values_tf = [0] * len(annotation)
    values_df = [0] * len(annotation)
    values_idf = [0] * len(annotation)
    for i in range(1, len(line), 4):
        line[i] = line[i].replace("\'", " ")
        line[i] = line[i].replace('"', '')
        line[i] = line[i].replace(',', ' ')
        values_tf[dict_annot[line[i]]] = (int(line[i + 1]))
        values_df[dict_annot[line[i]]] = (int(line[i + 2]))
        values_idf[dict_annot[line[i]]] = (float(line[i + 3]))
    annot_values_tf.append([line[0]] + values_tf)
    annot_values_df.append([line[0]] + values_df)
    annot_values_idf.append([line[0]] + values_idf)

final_annot_tf = list(map(list, zip(*annot_values_tf)))
final_annot_df = list(map(list, zip(*annot_values_df)))
final_annot_idf = list(map(list, zip(*annot_values_idf)))

# Dumping pickle file and saving the object for faster retrieval next time
pickle.dump(final_annot_tf , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_tf.p", "wb" ) )
pickle.dump(final_annot_df , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_df.p", "wb" ) )
pickle.dump(final_annot_idf , open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\save_idf.p", "wb" ) )

# Writing the schema into csv files for future access
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\TF_Img.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar = "'")
    for item in final_annot_tf:
        #Write item to outcsv
        writer.writerow(item)
        
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\DF_Img.csv", 'w', encoding = 'utf8', newline='') as outcsv: 
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar = "'")
    for item in final_annot_df:
        #Write item to outcsv
        writer.writerow(item)
        
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_2\\TF_IDF_Img.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar = "'")
    for item in final_annot_idf:
        #Write item to outcsv
        writer.writerow(item)