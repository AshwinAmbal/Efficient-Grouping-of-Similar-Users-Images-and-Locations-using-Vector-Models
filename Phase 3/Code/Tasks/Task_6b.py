"""
    The code below gives the implementation of the page rank algorithm to find the 'k' most relevent/
    significant/important images for a set of images and a user given 'k' as given by the page rank
    algorithm.
"""

import json
import xml.etree.ElementTree as ET
import os
import webbrowser
from collections import defaultdict

column_sim_model = json.load(open("C:\\MWDB Project\\Img_Img_Graph_k_7.json"))

image_ids = list()
for key in column_sim_model:
    image_ids.append(key)

def pagerank(Graph, alpha=0.85, personalization=None, 
			max_iter=100, tol=1.0e-6, nstart=None, 
			dangling=None): 
    """
    Return the PageRank of the nodes in the graph. 

    PageRank computes a ranking of the nodes in the graph G based on 
    the structure of the incoming links. It was originally designed as 
    an algorithm to rank web pages. 

    Parameters 
    ---------- 
    Graph : graph 
    Graph obtained from Task 1. 
    
    alpha : float, optional 
    Damping parameter for PageRank, default=0.85. 

    personalization: dict, optional 
    The "personalization vector" consisting of a dictionary with a 
    key for every graph node and nonzero personalization value for each node. 
    By default, a uniform distribution is used. 

    max_iter : integer, optional 
    Maximum number of iterations in power method eigenvalue solver. 

    tol : float, optional 
    Error tolerance used to check convergence in power method solver. 

    weight : key, optional 
    Edge data key to use as weight. If None weights are set to 1. 

    dangling: dict, optional 
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without 
    any outedges. The dict key is the node the outedge points to and the dict 
    value is the weight of that outedge. By default, dangling nodes are given 
    outedges according to the personalization vector (uniform if not 
    specified). This must be selected to result in an irreducible transition 
    matrix (see notes under google_matrix). It may be common to have the 
    dangling dict to be the same as the personalization dict. 

    Returns 
    ------- 
    pagerank : dictionary 
    Dictionary of nodes with PageRank as value 

    Notes 
    ----- 
    The eigenvector calculation is done by the power iteration method 
    and has no guarantee of convergence. The iteration will stop 
    after max_iter iterations or an error tolerance of 
    number_of_nodes(G)*tol has been reached. 

    The PageRank algorithm was designed for directed graphs but this 
    algorithm does not check if the input graph is directed and will 
    execute on undirected graphs by converting each edge in the 
    directed graph to two edges. 

	
    """
    if len(Graph) == 0: 
        return {} 

    N = len(Graph)

    # Choose fixed starting vector if not given 
    if nstart is None: 
        x = dict.fromkeys(list(Graph.keys()), 1.0 / N)
    else:
        # Normalized nstart vector 
        s = float(sum(nstart.values())) 
        x = dict((k, v / s) for k, v in nstart.items()) 
    if personalization is None: 
        # Assign uniform personalization vector if not given 
        p = dict.fromkeys(list(Graph.keys()), 1.0 / N)
    else: 
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())
    i = 0
    for _ in range(max_iter):
        i += 1
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0) 
        #danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
        for n in x:
			# this matrix multiply looks odd because it is 
			# doing a left multiply x^T=xlast^T*W 
            temp_sum = sum(b[1] for b in Graph[n])
            for nbr in Graph[n]:
                x[nbr[0]] += (xlast[n]/len(Graph[n])) * (nbr[1] / temp_sum)
        for n in x:
            x[n] = x[n] * alpha + (1 - alpha) * p[n]
		# check convergence, l1 norm 
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol: 
            print("Converged at ", i)
            return x
        
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
    
x_train = []
y_train = []
x_test = []
train_img = []

for pair in train_label:
    train_img.append(pair[0])
    y_train.append(pair[1])

# create a distinct labels set     
labels = list(set(y_train))

label_to_img = defaultdict(list)

scores = dict()

for img, label in zip(train_img, y_train):
	# assign the label to the image ID
    label_to_img[label].append(img)

cont = 'y'
for label in labels:
    imgid = label_to_img[label]
    person_dict = dict.fromkeys(image_ids, 0.0)
    
    for img in imgid:
        if img in person_dict:
            person_dict[img] = 1.0
        else:
            print("Image {} not found".format(img))

    # call the pagerank function 
    ppr = pagerank(column_sim_model, 0.85, personalization=person_dict, max_iter=100)
    sorted_by_val = sorted(ppr.items(), key=lambda kv: kv[1], reverse = True)
    for image in sorted_by_val:
        if image[0] == '1001205877':
            print(imgid, image[0], image[1], label)
        scores[(image[0],label)] = image[1]

# create a default dictionary to store the image labels
img_label = defaultdict(list)
for img in image_ids:
    temp_score = list()
    for label in labels:
        temp_score.append([img, label, scores[(img, label)]])
    temp_score = sorted(temp_score, key=lambda x:x[2], reverse=True)
    img_label[img].append([temp_score[0][1],temp_score[0][2]])
    i = 1
    while i < len(labels) and temp_score[i][2] == img_label[img][0][1]:
        img_label[img].append([temp_score[i][1],temp_score[i][2]])
        i += 1

# remove the sample image IDs from the test image IDs list  
test_img = list(set(image_ids)-set(train_img))

task = "Task 6b"

# create an html file to visualize the labeled results
f = open('C:\\MWDB Project\\Phase 3\\Code\\CSV\\{}.html'.format(task),'w')

message = """<html>
<head><title>{}</title></head>
<body>
<center>
<h1>{}</h1>
</center>
<h3>&nbsp&nbsp&nbsp&nbspGiven Images With Labels: </h3>
<br><div style="text-align:center; width:100%">""".format(task,task)

for i,(img,label) in enumerate(zip(train_img, y_train)):
        message = message + """<center><div class="img-with-text"  style="padding: 10px;margin:0 auto;float:left; display: inline-block; height: 300px; width:340px">
                        <img src="{}" height="250" width="350">
                        <p>{}&nbsp&nbsp&nbsp{}</p>
                        </div></center>""".format(img_to_path[img], img, label)
                        
message = message + """
</div> 
<div style="text-align:center; clear:left">
<h3 style="text-align:left">Classified Images With Labels as Given by PPR: </h3>"""

for i,img in enumerate(test_img):
        
        message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340px">
                        <img src="{}" height="250" width="350"><p>""".format(img_to_path[img])
        for im in img_label[img]:
            message = message + """{}&nbsp,&nbsp{}<br>""".format(im[0], im[1])
        message = message + """</p></div>"""
message = message + """
</div>
</body>
</html>"""
 
f.write(message)
f.close()

filename = 'C://MWDB Project//Phase 3//Code//CSV//' + '{}.html'.format(task)

# open the HTML file in the web browser
webbrowser.open_new_tab(filename)