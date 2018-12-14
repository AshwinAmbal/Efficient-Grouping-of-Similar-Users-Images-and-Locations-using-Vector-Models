import numpy as np  
import csv 
import pandas as pd  
from collections import Counter
import os
import json
import xml.etree.ElementTree as ET
import webbrowser

def train(X_train, y_train):
	return


def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(pd.to_numeric(x_test) - pd.to_numeric(X_train.iloc[i, :]))))
		# add it to list of distances
		distances.append([distance, i])
        
	# sort the list
	distances = sorted(distances)
	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# check if k larger than n
	if k > len(X_train):
		raise ValueError
		
	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test.iloc[i, :], k))

tree = ET.parse("C:\\MWDB Project\\devset_topics.xml")
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

# create a dictionary to hold the path to images
img_to_path = dict()
             
for loc in location_id:
    path ="C:\\MWDB Project\\devset\\img\\img\\{}".format(location_id[loc])
    dirListing = os.listdir(path)
    files=list()
    for file in dirListing:
        if file.replace(".jpg","") not in img_to_path:
            img_to_path[file.replace(".jpg","")] = path + "\\" + file

# dump the filepath information into a JSON file                        
json.dump(img_to_path, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json", 'w'))

img_to_path = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json"))
train_label = []

# read the input from the text file
sample = open("C:\\MWDB Project\\img_label_sample1.txt", "r", encoding = "utf8")
list_of_lines = sample.readlines()
for line in list_of_lines[2:]:
    words = line.split()
    train_label.append(words)    

# load the CM3x3 visual descriptors values
csvfile = open("C:\\MWDB Project\\CM3x3.csv" ,"r", encoding="utf8")
reader = csv.reader(csvfile, delimiter=',', quotechar = "'")
list_of_lines = list(reader)

loader_list = list(map(list, zip(*list_of_lines)))

# Create a DataFrame to hold the color model values
df = pd.DataFrame(loader_list)
df.columns = df.iloc[0]
df = df[1:]

x_train = []
y_train = []
x_test = []
train_img = []

for pair in train_label:
    train_img.append(pair[0])
    x_train.append(list(df[pair[0]]))
    y_train.append(pair[1])

# remove the sample image IDs from the test image IDs list  
test_img = list(set(df.columns)-set(train_img))

for column in test_img:
    x_test.append(list(df[column]))

# Convert the list into DataFrame
x_test = pd.DataFrame(x_test)
x_train = pd.DataFrame(x_train)
    
predictions = []
try:
	# call the k-nearest classifier
    kNearestNeighbor(x_train, y_train, x_test, predictions, 7)

    # store the predictions under y_pred
    y_pred = np.asarray(predictions)
except ValueError:
    print('Can\'t have more neighbors than training samples!!')


results = dict()
x_test_img = np.array(test_img).tolist()

for img, label in (zip(x_test_img, y_pred)):
	# store the labels under corresponding image ID
    results[img] = label
    
task = "Task 6a"

# create an html file to visualize the labeled results
f = open('C:\\MWDB Project\\Phase 3\\Code\\CSV\\{}.html'.format(task),'w')

message = """<html>
<head><title>{}</title></head>
<body>
<center>
<h1>{}</h1>
</center>
<h3>&nbsp&nbsp&nbsp&nbspGiven Images with labels: </h3>
<br><div style="text-align:center; width:100%">""".format(task,task)
    
for i, (img, label) in enumerate(zip(train_img, y_train)):
        message = message + """<center><div class="img-with-text"  style="padding: 10px;margin:0 auto;float:left; display: inline-block; height: 300px; width:340px">
                    <img src="{}" height="250" width="350">
                    <p>{}&nbsp&nbsp&nbsp{}</p>
                    </div></center>""".format(img_to_path[img], img, label)
                        
message = message + """
</div>
<div style="text-align:center; clear:left">
<h3 style="text-align:left">&nbsp&nbsp&nbsp&nbspMost Relevant Images Given by KNN: </h3>"""

for i,(img, label) in enumerate(results.items()):
        message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340px">
                        <img src="{}" height="250" width="350">
                        <p>{}&nbsp,&nbsp{}</p>
                        </div>""".format(img_to_path[img], img, label)

message = message + """
</div>
</body>
</html>"""
 
f.write(message)
f.close()

filename = 'C://MWDB Project//Phase 3//Code//CSV//' + '{}.html'.format(task)

# open the HTML file in the web browser
webbrowser.open_new_tab(filename)