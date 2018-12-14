"""
Description: The code below is used to extract required data from the file 
devset_textTermsPerUser.txt and write the required information in a database type
format into a csv file. The column header in the written file would signify the
user id's of the various users and the row headers would be annotations.
Eg:
    [Annotations UserID1 UserID2 UserID3]
    ["wicked"        ID1      ID2    ID3]
    [......          ....     ....   ...]

"""

import csv

# Opening file to read from
file1 = open("C:\\MWDB Project\\devset\\desctxt\\devset_textTermsPerUser.txt","r", encoding="utf8")
list_of_lines = file1.readlines()

#Reading all user id's into 'ids'
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
    
# Reading line by line from the file and writing the UserID and their corresponding
# tf, df and tf-idf values as rows and finally taking a transpose of the matrix
# formed to get the required database schema mentioned above.

annot_values_idf = [["Annotations"] + annotation]
for row in list_of_lines:
    line = row.split()
    flag = 0
    values_idf = [0] * len(annotation)
    for i in range(1, len(line), 4):
        line[i] = line[i].replace("\'", " ")
        line[i] = line[i].replace('"', '')
        line[i] = line[i].replace(',', ' ')
        values_idf[dict_annot[line[i]]] = (float(line[i + 3]))
    annot_values_idf.append([line[0]] + values_idf)

final_annot_idf = list(map(list, zip(*annot_values_idf)))


with open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 1_2\\TF_IDF_User.csv", 'w', encoding = 'utf8', newline='') as outcsv:   
    #configure writer to write standard csv file
    writer = csv.writer(outcsv, delimiter=',', quotechar = "'")
    for item in final_annot_idf:
        #Write item to outcsv
        writer.writerow(item)