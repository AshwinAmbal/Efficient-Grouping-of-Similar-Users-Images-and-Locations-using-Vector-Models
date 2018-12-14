"""
    The code below gives the implementation of the page rank algorithm to find the 'k' most relevent/
    significant/important images for a set of images and a user given 'k' as given by the page rank
    algorithm.
"""

import json
import xml.etree.ElementTree as ET
import os
import webbrowser

g = int(input("Enter the 'k' value of Graph to use: "))

column_sim_model = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(g)))

image_ids = list()
for key in column_sim_model:
    image_ids.append(key)

def pagerank(Graph, alpha=0.85, personalization=None, 
			max_iter=100, tol=1.0e-6): 
    """
    Return the PageRank of the nodes in the graph. 

    Parameters 
    ---------- 
    Graph : graph of image-image similarity 
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
    x = dict.fromkeys(list(Graph.keys()), 1.0 / N)
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
        for n in x:
            temp_sum = sum(b[1] for b in Graph[n])
            for nbr in Graph[n]:
                x[nbr[0]] += (xlast[n]/len(Graph[n])) * (nbr[1] / temp_sum)
        for n in x:
            x[n] = x[n] * alpha + (1 - alpha) * p[n]
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol: 
            print("Converged at ", i)
            return x

# Storing paths of images in respective locations to visualize later        
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

img_to_path = dict()
             
for loc in location_id:
    path ="C:\\MWDB Project\\devset\\img\\img\\{}".format(location_id[loc])
    dirListing = os.listdir(path)
    files=list()
    for file in dirListing:
        if file.replace(".jpg","") not in img_to_path:
            img_to_path[file.replace(".jpg","")] = path + "\\" + file
                        
json.dump(img_to_path, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json", 'w'))

img_to_path = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json"))

cont = 'y'
# Calculating PPR and visualizing the images relevant to the personalized images.
while cont.lower() == 'y':
    k = int(input("Enter the value of k: "))
    K = int(input("Enter number of images to find similarity with: "))
    imgid = list()
    for i in range(0, K):
    	imgid.append(str(input("Enter Image ID {}: ".format(i+1))))
    person_dict = dict.fromkeys(image_ids, 0.0)
    
    for img in imgid:
        if img in person_dict:
            person_dict[img] = 1.0
        else:
            print("Image {} not found".format(img))
    ppr = pagerank(column_sim_model, 0.75, personalization=person_dict, max_iter=100)
    sorted_by_val = sorted(ppr.items(), key=lambda kv: kv[1], reverse = True)[:k]
    print(sorted_by_val)
    
    task = "Task 4"
    f = open('C:\\MWDB Project\\Phase 3\\Code\\CSV\\{}.html'.format(task),'w')
    
    message = """<html>
    <head><title>{}</title></head>
    <body>
    <center>
    <h1>{}</h1>
    </center>
    <h3>&nbsp&nbsp&nbsp&nbspGiven Images: </h3>
    <br><div style="text-align:center; width:100%">""".format(task,task)
    
    for i,img in enumerate(imgid):
            message = message + """<center><div class="img-with-text"  style="padding: 10px;margin:0 auto;float:left; display: inline-block; height: 300px; width:340px">
                            <img src="{}" height="250" width="350">
                            <p>{}</p>
                            </div></center>""".format(img_to_path[img], img)
                            
    message = message + """
    </div>
    <div style="text-align:center; clear:left">
    <h3 style="text-align:left">&nbsp&nbsp&nbsp&nbspMost Relevant Images Given by PPR: </h3>"""
    
    for i,img in enumerate(sorted_by_val):
            message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340px">
                            <img src="{}" height="250" width="350">
                            <p>{}&nbsp,&nbsp{}</p>
                            </div>""".format(img_to_path[img[0]], img[0], img[1])
    
    message = message + """
    </div>
    </body>
    </html>"""
     
    f.write(message)
    f.close()
    
    filename = 'C://MWDB Project//Phase 3//Code//CSV//' + '{}.html'.format(task)
    webbrowser.open_new_tab(filename)
    cont = input("Do you want to continue? (Y/N): ")