"""
    Description: Location-Location Similarity matrix is formed which would 
    be a symmetric matrix. This is essentially running "Task_5.py" for each 
    location and finding the similarity score for that location with every 
    other location. The resultant matrix is saved to "Similarity_Matrix.csv" 
    file in the Code/CSV folder. SVD is then performed on this matrix and 
    the top k-latent semantics is listed along with the corresponding 
    term-weight pairs.
"""

import csv
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
#from sklearn import preprocessing

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
models = ['CM', 'CN' , 'CM3x3' , 'CN3x3' , 'CSD' , 'GLRLM' , 'GLRLM3x3' , 'HOG' , 'LBP' , 'LBP3x3']
#models = ['GLRLM3x3' , 'LBP3x3']

#Building the location-location similarity matrix (Using Cosine Similarity)
def build_sim_matrix():
    arr = [[1] * (len(location_id) + 1) for _ in range(1, len(location_id) + 2)]
    for loc in range(1, len(location_id) + 1):
        column_sim_model = defaultdict(list)
        for model in models:
            csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5_Img\\{}.csv".format(location_id[loc] + " " + model),"r", encoding="utf8")
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
            it = 0
            for number in file_numbers:
                if number <= loc:
                    column_sim_sum[c] = [0, 0] + [0]
                    c += 1
                    continue
                it += 1
                list_of_lines = list()
                csvfile = open("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 4_5_Img\\{}.csv".format(location_id[number] + " " + model),"r", encoding="utf8")
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
            sum_temp = sorted(column_sim_sum, key = lambda x : x[2], reverse = True)
            column_sim_model[model].append(sum_temp)
        for loc_num in range(loc + 1, len(location_id) + 1):
            sum_of_model_sim = 0
            for key, value in column_sim_model.items():
                for i, req_list in enumerate(value[0]):                    
                    if req_list[0] == loc_num:
                        sum_of_model_sim += req_list[2]
            arr[loc][loc_num] = sum_of_model_sim / len(models)
            arr[loc_num][loc] = sum_of_model_sim / len(models)
        print("Done with Loc: ", loc)
        print("Number of Iterations: ", it)
        #print(arr)
    return arr


# Function to read the location-location similarity matrix
def get_sim_matrix():
    df = pd.DataFrame()
    df = df.from_csv("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Similarity_Matrix.csv", encoding = 'utf8')
    return df

#arr = build_sim_matrix()

#loc_labels = [location_id[x] for x in range(1, len(location_id) + 1)]
#df = pd.DataFrame(arr)
#df = df.drop(0, 1)
#df = df[1:]
#df.columns = loc_labels
#df.index = loc_labels
# Saving to csv for faster retrieval in future.
#df.to_csv("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Similarity_Matrix.csv", index = True, encoding = 'utf8')

df = pd.DataFrame()
df = get_sim_matrix()

k = int(input("Enter the value of k "))

# Performing SVD on the loc-loc similarity matrix
scaled_data = preprocessing.scale(df.T)
svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
svd.fit(scaled_data)
svd_data=svd.transform(scaled_data)

print("\n")
print("Latent Semantic Term - Weight Pairs: " )
for p, row in enumerate(svd.components_):
    print("k = ", p, "=>", row)

new_df=pd.DataFrame(svd_data)
new_df=new_df.T
new_df.columns = df.columns

# Saving to csv for faster retrieval in future.
new_df.to_csv("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Similarity_Matrix SVD.csv", index = True, encoding = 'utf8')