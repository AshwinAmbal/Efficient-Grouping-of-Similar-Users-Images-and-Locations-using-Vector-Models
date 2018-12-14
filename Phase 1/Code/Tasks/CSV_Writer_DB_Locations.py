# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:42:06 2018

@author: AshwinAmbal

Description: The code below is used to extract required data from the file 
devset_textTermsPerPOI.txt and write the required information in a database type
format into a csv file. The column header in the written file would signify the
user id's of the various users and the row headers would be annotations.
Eg:
    [Annotations  LocID1 LocID2 LocID3]
    ["wicked"      ID1      ID2    ID3]
    [......        ....     ....   ...]
"""

import csv

# Opening file to read from
file1 = open("C:\\MWDB Project\\devset\\desctxt\\devset_textTermsPerPOI.txt","r", encoding="utf8")
list_of_lines = file1.readlines()
ids = list()

#Reading all image id's into 'ids'
for row in list_of_lines:
    loc_id = ""
    words = row.split()
    for word in words:
        if '"' not in word:
            loc_id += word + " "
        else:
            loc_id = "_".join(loc_id.split())
            break
    ids.append(loc_id)

# Converting existing file to one in which the first set of words before the 
# annotations are replaced with _ between those words for uniform access 
# devset_textTermsPerPOIEdited.csv ===> devset_textTermsPerPOI.csv
new_lines = list()
for i, row in enumerate(list_of_lines):
    w = 0
    words = row.split()
    new_line = list()
    for word in words:
        if '"' in word:
            break
        w += 1
    new_line.append(ids[i])
    for j in range(w, len(words)):
        new_line.append(words[j])
    new_lines.append(new_line)


with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\devset_textTermsPerPOIEdited.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter = ',', quotechar = "'")
    for item in new_lines:
        #Write item to outcsv
        writer.writerow(item)

list_of_lines = list()
csvfile = open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\devset_textTermsPerPOIEdited.csv","r", encoding="utf8")
reader = csv.reader(csvfile, delimiter=',', quotechar = "'") 
for row in reader:
    list_of_lines.append(row)

# Reading all annotations into 'annotation'
annotation = list()
for line in list_of_lines:
    for i in range(1, len(line), 4):
        line[i] = line[i].replace("\'", " ")
        line[i] = line[i].replace('"', '')
        annotation.append(line[i])

annotation = list(set(annotation))

# Making a dictionary of annotation mapping to the index in which it occurs in the
# list 'annotation'
dict_annot = dict()
for i, annot in enumerate(annotation):
    dict_annot[annot] = i

# Reading line by line from the file and writing the UserID and their corresponding
# tf, df and tf-idf values as rows and finally taking a transpose of the matrix
# formed to get the required database schema mentioned above.
annot_values_tf = [["Annotations"] + annotation]
annot_values_df = [["Annotations"] + annotation]
annot_values_idf = [["Annotations"] + annotation]
for line in list_of_lines:
    flag = 0
    values_tf = [0] * len(annotation)
    values_df = [0] * len(annotation)
    values_idf = [0] * len(annotation)
    for i in range(1, len(line), 4):
        line[i] = line[i].replace("\'", " ")
        line[i] = line[i].replace('"', '')
        values_tf[dict_annot[line[i]]] = (int(line[i + 1]))
        values_df[dict_annot[line[i]]] = (int(line[i + 2]))
        values_idf[dict_annot[line[i]]] = (float(line[i + 3]))
    annot_values_tf.append([line[0]] + values_tf)
    annot_values_df.append([line[0]] + values_df)
    annot_values_idf.append([line[0]] + values_idf)

final_annot_tf = list(map(list, zip(*annot_values_tf)))
final_annot_df = list(map(list, zip(*annot_values_df)))
final_annot_idf = list(map(list, zip(*annot_values_idf)))

# Writing the schema into csv files for future access
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\TF_Loc.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',')
    for item in final_annot_tf:
        #Write item to outcsv
        writer.writerow(item)
        
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\DF_Loc.csv", 'w', encoding = 'utf8', newline='') as outcsv: 
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',')
    for item in final_annot_df:
        #Write item to outcsv
        writer.writerow(item)
        
with open("C:\\MWDB Project\\Phase 1\\Code\\CSV\\Task_3\\TF_IDF_Loc.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',')
    for item in final_annot_idf:
        #Write item to outcsv
        writer.writerow(item)