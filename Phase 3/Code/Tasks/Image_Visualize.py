# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:46:35 2018

@author: aambalav
"""

import xml.etree.ElementTree as ET
import os
import json
import webbrowser

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
sim =12334567        
for loc in location_id:
    path ="C:\\MWDB Project\\devset\\img\\img\\{}".format(location_id[loc])
    dirListing = os.listdir(path)
    files=list()
    for file in dirListing:
        if file.replace(".jpg","") not in img_to_path:
            img_to_path[file.replace(".jpg","")] = path + "\\" + file
                        
json.dump(img_to_path, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json", 'w'))

img_to_path = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Path.json"))

images = ['3103006230','2718956990','12497387424','1675682410','1832536520', '2907638727', '1832538318', '223803485']
task = "Task 3"
f = open('C:\\MWDB Project\\Phase 3\\Code\\CSV\\{}.html'.format(task),'w')

message = """<html>
<head><title>{}</title></head>
<body>
<center>
<h1>{}</h1>
</center>
<br><div style="text-align:center">""".format(task,task)

for i,img in enumerate(images):
        message = message + """<div class="img-with-text"  style="padding: 10px;float:left; display: inline-block; height: 300px; width:340">
                        <img src="{}" height="250" width="350">
                        <p>{}&nbsp,&nbsp{}</p>
                        </div>""".format(img_to_path[img], img, sim)

message = message + """
</div>
</body>
</html>"""
 
f.write(message)
f.close()

filename = 'C://MWDB Project//Phase 3//Code//CSV//' + '{}.html'.format(task)
webbrowser.open_new_tab(filename)