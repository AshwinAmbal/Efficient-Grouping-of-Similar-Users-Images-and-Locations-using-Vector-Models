"""
    Description: Given a visual descriptor model and a value 'k', the 
    image feature space of the given color model for each location is 
    reduced along the color model values and the corresponding term-weight 
    pairs are listed. For example, for an image in CM model, there will be 
    9 values and given k = 5 those 9 values will be reduced to 5 values 
    using PCA, SVD or LDA as inputted by the user. An Image ID is then 
    inputted by the user for which the similarity needs to be computed. 
    This image is compared with all other images in the dataset and the 
    most similar 5 images and their corresponding locations are returned. 
    To find 5 most similar locations to the location of the given image id, 
    the average of similarity scores of the given image compared with all 
    images of a particular location is calculated for each of the 
    locations in the devset. The top 5 average scores and the 
    corresponding location is returned.
"""

import csv
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
#import matplotlib.pyplot as plt\

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
     
cont = 'y'          
image_to_loc = dict()  

red = int(input("Enter the reduction that you would like to perform \n 1.PCA 2.SVD 3.LDA : " ))


while(cont.lower() == 'y'):
    model = input("Enter the model to be considered: ")
    k = int(input("Enter the value of k "))
    
    # Perform PCA
    if(red == 1):
        for i, num in enumerate(location_id):
            csvfile = open("C:\\MWDB Project\\devset\\descvis\\img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            
            for column in df.columns:
                df[column] = pd.to_numeric(df[column])
                image_to_loc[column] = num
                
            scaled_data = preprocessing.scale(df.T)
            pca = PCA(n_components = k)
            #pca = PCA(0.95)
            pca.fit(scaled_data)
            pca_data = pca.transform(scaled_data)
            print()
            print("PCA for Location: ", location_id[num])
            print("\nTotal Variance Accounted for: ", (pca.explained_variance_ratio_* 100).sum())
            
            print("\n")
            print("Latent Semantic Term - Weight Pairs: " )
            for p, row in enumerate(pca.components_):
                print("k = ", p, " ", row)
                print()
            
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
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)
    
    # Perform SVD
    elif(red == 2):
        for i, num in enumerate(location_id):
            csvfile = open("C:\\MWDB Project\\devset\\descvis\\img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            
            for column in df.columns:
                df[column] = pd.to_numeric(df[column])
                image_to_loc[column] = num
                
            scaled_data = preprocessing.scale(df.T)
            
            
            svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
            svd.fit(scaled_data)
            svd_data=svd.transform(scaled_data)
            print()
            
            print("\n")
            print("Latent Semantic Term - Weight Pairs: " )
            for p, row in enumerate(svd.components_):
                print("k = ", p, " ", row)
                print()
            
                
            new_df=pd.DataFrame(svd_data)
            new_df=new_df.T
            new_df.columns = df.columns
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)
        
    # Perform LDA
    else:
        for i, num in enumerate(location_id):
            csvfile = open("C:\\MWDB Project\\devset\\descvis\\img\\{}.csv".format(location_id[num] + " " + model))
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            
            loader_list = list(map(list, zip(*list_of_lines)))
            
            df = pd.DataFrame(loader_list)
            df.columns = df.iloc[0]
            df = df[1:]
            for column in df.columns:
                image_to_loc[column] = num
                
            m = 10000000
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                if m > float(df[col].min()):
                    #print(m,float(df[col].min()))
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
            
            file = "C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(location_id[num] + " " + model)
            list_of_lines = new_df.T.values.tolist()
            with open(file, 'w', encoding = 'utf8', newline='') as outcsv:   
                writer = csv.writer(outcsv, delimiter=',')
                for img, line in zip(new_df.columns, list_of_lines):
                    writer.writerow([img] + line)
    
    # Using cosine similarity metric to find similarity between location 
    # based on the image id given by user
    inner_cont = 'y'
    
    while(inner_cont.lower() == 'y'):
        image_id = input("Enter the Image ID: ")
        loc = location_id[image_to_loc[image_id]]
        print("\nCorresponding Location of Image: ", loc)
        
        # Finding Image Similarity
        csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(loc + " " + model),"r", encoding="utf8")
        reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
        list_of_lines = list(reader)
        for line in list_of_lines:
            if image_id == line[0]:
                image_df = pd.DataFrame(line)
        image_df.columns = image_df.iloc[0]
        image_df = image_df[1:]
        
        print("\n")
    
        sim = list()
        file_numbers = list(range(1, len(location_id) + 1))  
        file_numbers.remove(image_to_loc[image_id])
        for number in file_numbers:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            loader_list = list(map(list, zip(*list_of_lines)))
            comp_df = pd.DataFrame(loader_list)
            comp_df.columns = comp_df.iloc[0]
            comp_df = comp_df[1:]
            for column in comp_df:
                if column != image_id:
                    sim.append([column, 1 - cosine(pd.to_numeric(image_df[image_id]), pd.to_numeric(comp_df[column])), location_id[number]])
                
        results = sorted(sim,  key = lambda x : x[1], reverse=True)[:5]
                
        print("Most Similar 5 Images for ID = ", image_id)
        print("\nUSING COSINE SIMILARITY:")
        for row in results:
            print(row)
        print()
        
        sim = list()
        results = list()
        file_numbers = list(range(1, len(location_id) + 1))   
        file_numbers.remove(image_to_loc[image_id])
        for number in file_numbers:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 3\\{}.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
            reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
            list_of_lines = list(reader)
            loader_list = list(map(list, zip(*list_of_lines)))
            comp_df = pd.DataFrame(loader_list)
            comp_df.columns = comp_df.iloc[0]
            comp_df = comp_df[1:]
            for num in range(0, len(comp_df.columns)):
                sim.append(1 - cosine(pd.to_numeric(image_df[image_id]), pd.to_numeric(comp_df.iloc[ : ,num])))
            sum_val = sum(sim) / len(comp_df.columns)
            results.append([number, location_id[number], sum_val])
        final_result = sorted(results,  key = lambda x : x[2], reverse=True)[:5]
                
        print("Most Similar 5 Locations for Img ID = ", image_id, " and Location ID: ", loc)
        print("\nUSING COSINE SIMILARITY:")
        for row in final_result:
            print(row)   
        print()
        
        inner_cont = input("Do you want to check another Image ID with same color model? (Y/N): ")
    cont = input("Do you want to continue? (Y/N): ")