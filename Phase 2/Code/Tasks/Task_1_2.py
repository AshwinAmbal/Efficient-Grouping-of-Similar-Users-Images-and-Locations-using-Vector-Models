"""
    Description: The code first reduces the terms to k-latent semantics 
    using PCA, SVD or LDA as inputted by the user and the corresponding 
    term-weight pairs are listed. The code then extracts 'k' similar users, 
    images or locations (based on user input) for a given user, image or 
    location id in the reduced space based on the tf-idf values.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
                for topic_idx, topic in enumerate(model.components_):
                    print ("k = ", topic_idx)
                    print ([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
                    
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
            
cont = 'Y'

while(cont.lower() == 'y'):
    red = int(input("Enter the reduction that you would like to perform \n 1.PCA 2.SVD 3.LDA : " ))
    file = input("Enter the Term Space (User, Img, Loc) to consider: ")
    df = pd.read_csv("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 1_2\\TF_IDF_{}.csv".format(file))
    
    # Perform PCA
    if(red == 1):
        k = int(input("Enter the value of k: "))
        pca_list=list(df.iloc[:,0])
        df = df.set_index('Annotations')
        
        scaled_data = preprocessing.scale(df.T)

        pca = PCA(n_components = k)
        #pca = PCA(0.95)
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        
        print("Total Variance Accounted for: ", (pca.explained_variance_ratio_* 100).sum())
        
        print("\n")
        print("Latent Semantic Term - Weight Pairs: " )
        print()
        display_topics(pca, pca_list, 10)
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
    
    # Perform SVD
    elif(red == 2):
        k = int(input("Enter the value of k: "))
        svd_list=list(df.iloc[:,0])
        df = df.set_index('Annotations')
        scaled_data = preprocessing.scale(df.T)
        
        svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
        svd.fit(scaled_data)
        svd_data=svd.transform(scaled_data)
        
        print("\n")
        print("Latent Semantic Term - Weight Pairs: " )
        print()
        display_topics(svd, svd_list, 10)
        print()
         
        new_df=pd.DataFrame(svd_data)
        new_df=new_df.T
        new_df.columns = df.columns
        
    # Perform LDA
    else:
        k = int(input("Enter the value of k: "))
        
        lda_list=list(df.iloc[:,0])
        df = df.set_index("Annotations")
        lda_d = df.T
        lda = LatentDirichletAllocation(n_topics= k, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
        lda.fit(lda_d)
        X = lda.transform(lda_d)
        new_df = pd.DataFrame(X)
        new_df = new_df.T
        new_df.columns = df.columns
        #for l, row in enumerate(lda.components_):
        #    print("topic = ", l, "=>", row)
        print("\n")
        print("Latent Semantic Term - Weight Pairs: " )
        print()
        display_topics(lda, lda_list, 10)
    
    # Using cosine similarity metric to find similarity between users, img
    # and location as given by user
    sim = list()
    print("\n")
    if(file == "User"):
        id_sim = input("Enter the User ID: ")
    elif(file == "Img"):
        id_sim = input("Enter the Image ID: ")
    else:
        id_sim = location_id[int(input("Enter the Location ID: "))]
    
    for column in new_df:
        if column != id_sim:
            sim.append([column, 1 - cosine(pd.to_numeric(new_df[id_sim]), pd.to_numeric(new_df[column]))])
        
    results = sorted(sim,  key = lambda x : x[1], reverse=True)[:5]
        
    print("\n\nMost Similar 5 Entities for ID = ", id_sim)
    print("\nUSING COSINE SIMILARITY:")
    for row in results:
        print(row)   
    print()
    
    cont = input("Do you want to continue? (Y/N) ")