"""
    Description: Given a value 'k', the image feature space of all color 
    models for each location is reduced along the color model values and the 
    corresponding term-weight pairs are listed. The code is then used to 
    extract 5 similar locations given a location id. For each match, the 
    code also lists the overall matching score cumulative of all the models. 
    It also lists the individual contributions of the 10 color models.
"""

import csv
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.spatial.distance import cosine
from collections import defaultdict
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
#import matplotlib.pyplot as plt

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

# Models to be considered while performing location-location similarity
models = ['CM3x3', 'CN3x3', 'CSD', 'GLRLM3x3', 'HOG', 'LBP3x3']

# Get input of which reduction to use
red = int(input("Enter the reduction that you would like to perform \n 1.PCA 2.SVD 3.LDA : " ))
k = int(input("Enter the value of k "))

# Perform PCA
if(red == 1):
    for i, num in enumerate(location_id):
        for model in models:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5_Img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            
            for column in df.columns:
                df[column] = pd.to_numeric(df[column])
            
            scaled_data = preprocessing.scale(df.T)
            pca = PCA(n_components = k)
            #pca = PCA(0.95)
            pca.fit(scaled_data)
            pca_data = pca.transform(scaled_data)
            
            print()
            print("PCA for Location: ", location_id[num], " Model: ", model)
            print("\nTotal Variance Accounted for: ", (pca.explained_variance_ratio_* 100).sum())
            
            print("\n")
            print("Latent Semantic Term - Weight Pairs: " )
            for p, row in enumerate(pca.components_):
                print("k = ", p, "=>", row)
            
            per_var = np.round(pca.explained_variance_ratio_* 100, decimals = 1)
            labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
            
            #plt.bar(range(1, len(per_var) + 1), per_var, tick_label = labels)
            #plt.ylabel('Percentage of Explained Variance')
            #plt.xlabel('Principal Component')
            #plt.title('Scree Plot')
            #plt.show()
            
            print("\n")
            print("Strength of the ", k, " Latent Semantics: " )
            for label, var in zip(labels, per_var):
                print(label, var)
            
            new_df = pd.DataFrame(pca_data)
            new_df = new_df.T
            new_df.columns = df.columns
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)

# Perform SVD
elif(red == 2):
    for i, num in enumerate(location_id):
        for model in models:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5_Img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            
            for column in df.columns:
                df[column] = pd.to_numeric(df[column])
            
            scaled_data = preprocessing.scale(df.T)
            svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
            svd.fit(scaled_data)
            svd_data=svd.transform(scaled_data)
            
            print()
            
            print("\n")
            print("Latent Semantic Term - Weight Pairs: " )
            for p, row in enumerate(svd.components_):
                print("k = ", p, "=>", row)
            
            new_df = pd.DataFrame(svd_data)
            new_df = new_df.T
            new_df.columns = df.columns
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)

# Perform LDA
else:
    for i, num in enumerate(location_id):
        for model in models:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5_Img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            image_to_loc = dict()
            for column in df.columns:
                image_to_loc[column] = num
                
            m = 10000000
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                if m>float(df[col].min()):
                    m = float(df[col].min())
            for col in df.columns:
                df[col] = -m+pd.to_numeric(df[col])

            lda_d = df.T
            lda = LatentDirichletAllocation(n_topics= k, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
            lda.fit(lda_d)
            X = lda.transform(lda_d)
            new_df = pd.DataFrame(X)
            new_df = new_df.T
            new_df.columns = df.columns
            
            print()
            print("LDA for Location: ", location_id[num])
            
            print("\n")
            print("Latent Semantic Term - Weight Pairs: " )
            
            for l, row in enumerate(lda.components_):
                print("topic = ", l, "=>", row)
                print()
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)
            
# Using cosine similarity metric to find similarity between location over all
# color models based on the location id given by user
inner_cont = 'y'
while(inner_cont.lower() == 'y'):
    loc = int(input("Enter the Location ID: "))
    column_sim_model = defaultdict(list)
    for model in models:
        csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5\\{}.csv".format(location_id[loc] + " " + model),"r", encoding="utf8")
        reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
        list_of_lines = list(reader)
        
        loader_list = list(map(list, zip(*list_of_lines)))
        
        df = pd.DataFrame(loader_list)
        df.columns = df.iloc[0]
        df = df[1:]
        
        file_numbers = list(range(1, len(location_id) + 1))
        file_numbers.remove(loc)
        column_sim_sum = [0] * (len(location_id) - 1)
        c = 0
        for number in file_numbers:
            list_of_lines = list()
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5\\{}.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            loader_list = list(map(list, zip(*list_of_lines)))
            comp_df = pd.DataFrame(loader_list)
            comp_df.columns = comp_df.iloc[0]
            comp_df = comp_df[1:]
            img_to_process = list(df.columns)
            second_to_first_sim = defaultdict(list)
            while img_to_process != []:
                column1 = img_to_process.pop()
                sim = list()
                temp_sim = list()
                for column2 in comp_df.columns:
                    temp_sim = [column1, column2, 1 - cosine(pd.to_numeric(df[column1]), pd.to_numeric(comp_df[column2]))]
                    sim.append(temp_sim)
                k = 0
                most_sim_img = sorted(sim, key = lambda x : x[2], reverse = True)[k]
                flag = 0
                while flag == 0:
                    if(most_sim_img[1] in second_to_first_sim):
                        if(second_to_first_sim[most_sim_img[1]][1] < most_sim_img[2]):
                            img_to_process.append(second_to_first_sim[most_sim_img[1]][0])
                            second_to_first_sim[most_sim_img[1]] = [most_sim_img[0], most_sim_img[2]]
                            flag = 1
                        else:
                            k += 1
                            if k == len(sim):
                                break
                            most_sim_img = sorted(sim, key = lambda x : x[2], reverse = True)[k]
                    else:
                        second_to_first_sim[most_sim_img[1]] = [most_sim_img[0], most_sim_img[2]]
                        flag = 1
            sum_val = 0
            for key, val in second_to_first_sim.items():
                sum_val += val[1]
            column_sim_sum[c] = [number, location_id[number]] + [sum_val]
            c += 1
        sum_temp = sorted(column_sim_sum, key = lambda x : x[2], reverse = True)[:5]
        column_sim_model[model].append(sum_temp)
    count_most_occ = dict()
    
    for index in range(1, len(location_id) + 1):
        count_most_occ[index] = 0
    
    for key, value in column_sim_model.items():
        for i, req_list in enumerate(value[0]):
            count_most_occ[req_list[0]] += (5-i)
    
    sorted_by_value = sorted(count_most_occ.items(), key=lambda kv: kv[1], reverse = True)[:5]
    
    print("\n\n Similarity score for location id: ", loc, " location name: ", location_id[loc])
    
    for item in sorted_by_value:
        print("\n", item[0], location_id[item[0]],"(", item[1], "/", 5 * len(models), ")")
        for model, value in column_sim_model.items():
            for index, tuple_row in enumerate(value[0]):
                if(tuple_row[0] == item[0]):
                    print("Model: ", model, ", Similarity Score: ", tuple_row[2], ", Rank: ", index + 1)
    print("\n")
    inner_cont = input("Do you want to check for another location? (Y/N): ")