"""
    Description: The xml files in devset/xml gives a grouping of users, 
    images, locations and corresponding tags common to all three. We use 
    these values to construct a tensor with loc x user x images which would 
    be a tensor of dimension 30 x 530 x 8912. This tensor would be very 
    sparse. And as specified by the task, given a value 'k', we perform 
    rank-k CP-Decomposition on this tensor and display the corresponding 
    factor matrices. We then perform K-means clustering with value 'k' 
    ('k' being the same inputted 'k' as before) on the three matrices to 
    form k-groups of non-overlapping user, image and locations as required 
    by the question.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
import tensorly as tl
import nltk
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans

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

# Reading the xml file for getting related user-loc-img
loc_map = defaultdict(list)
for number in location_id:
    tree = ET.parse("C:\\MWDB Project\\devset\\xml\\{}.xml".format(location_id[number]))
    root = tree.getroot()
    loc_map[num] = list()
    for item in root.findall('./photo'):
        loc_map[number].append([item.attrib['id'], item.attrib['userid'], len(nltk.word_tokenize(item.attrib['tags']))])
        #print(number)

# Forming user-index dictionary and vice versa to populate the appropriate
# space in the tensor
user_df = pd.DataFrame()
user_df = pd.read_csv("C:\\MWDB Project\\Phase 2\\Code\\CSV\\Task 1_2\\TF_IDF_User.csv")
user_to_idx = dict()
idx_to_user = dict()
k = 0
for user in user_df.columns:
    if user == 'Annotations':
        continue
    user_to_idx[user] = k
    idx_to_user[k] = user
    k += 1

# Forming img-index dictionary and vice versa to populate the appropriate
# space in the tensor
dist_list = list()
img_to_idx = dict()
idx_to_img = dict()
k = 0
for key, pairs in loc_map.items():
    for pair in pairs:
        dist_list.append(pair[0])
dist_list = set(dist_list)
len(dist_list)
for img in dist_list:
    img_to_idx[img] = k
    idx_to_img[k] = img
    k += 1

# Setting the dimensions for the tensor.
temp_arr = [0] * len(location_id) * len(user_to_idx) * len(img_to_idx)
arr = np.array(temp_arr).reshape(len(location_id), len(user_to_idx), len(img_to_idx))

# Building the tensor
for key, pairs in loc_map.items():
    for pair in pairs:
        arr[key - 1][user_to_idx[pair[1]]][img_to_idx[pair[0]]] = pair[2]

X = tl.tensor(arr)
X.shape
k = int(input("Enter the value of k: "))

# Performing CP-Decomposition on the tensor built.
factors = parafac(X, rank=k)

# Displaying the dimensions of the 3 factors
[f.shape for f in factors]


# Performing K-Means on the Location Factor matrix
kmeans = KMeans(n_clusters = k)
Z = pd.DataFrame(factors[0])
kmeans.fit(Z)
y_kmeans = kmeans.predict(Z)
k_groups_loc = defaultdict(list)
for loc, clust in enumerate(y_kmeans):
    k_groups_loc[clust].append(location_id[loc + 1])

for group, value in k_groups_loc.items():
    print("\n\nk = ", group, " => ", value)


# Performing K-Means on the User Factor matrix
Z = pd.DataFrame(factors[1])
kmeans.fit(Z)
y_kmeans = kmeans.predict(Z)
k_groups_user = defaultdict(list)
for loc, clust in enumerate(y_kmeans):
    k_groups_user[clust].append(idx_to_user[loc])

for group, value in k_groups_user.items():
    print("\n\nk = ", group, " => ", value)

# Performing K-Means on the Img-Factor matrix
Z = pd.DataFrame(factors[2])
kmeans.fit(Z)
y_kmeans = kmeans.predict(Z)
k_groups_img = defaultdict(list)
for loc, clust in enumerate(y_kmeans):
    k_groups_img[clust].append(idx_to_img[loc])

for group, value in k_groups_img.items():
    print("\n\nk = ", group, " => ", value)