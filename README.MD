MWDB Project - Fall'18 - Arizona State University

Download the MWDB Project folder to the C:/ Drive so that all file paths are supported.

The codebase is used for finding similarity between users, images and locations from features extracted from a set of images in flickr as given in the link: http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5

Check the detailed description of tasks in the pdf file attached with the project. The task code SHOULD be run in the order described for correct execution.

The dataset describes features describing Images taken in Locations that are annotated manually by Users.
There are two types of features: 
1. Textual Descriptors which consist of TF, DF and IDF values for each annotation grouped under user, image or location category.
2. Visual Descriptors which consists of 10 Color Models, namely, CM, CM3x3, CN, CN3x3,CSD,GLRLM,GLRLM3x3, HOG,LBP, LBP3x3 which give different sets of values for each image.

There are a total of 30 Locations. Each location has around 285 - 300 images. There are 10 Color models for each location which give different values for the images of that location.

The MWDB Project/Phase 1 folder works on finding similarity between users, images and locations using textual features such as TF, DF, TF-IDF and visual descriptors such as the ones specified above. Each task in the phase works off the knowledge obtained in the previous task.

The MWDB Project/Phase 2 is used for performing dimensionality reduction on the existing Object-Feature space using different techniques like PCA, SVD and LDA and finding similarity between users, images and locations from features extracted from a set of images.

The MWDB Project/Phase 3 is used for experimenting with graph analysis, clustering, indexing and classification.

Detailed information about instructions to run can be found in the README file inside the Phase 1 and Phase 2 folders.

The MWDB Project/devset folder contains the text descriptor and the visual descriptor features. (Should be downloaded from the URL specified above)

Description of items in MWDB Project/devset Folder:
	1. devset/desctxt- Has Textual Descriptors (TF, DF, TF-IDF) for grouped for Image, Location and Users.
	2. devset/descvis- Has Visual Descriptors for images in locations which is the output of 10 different color models(CM, CM3x3, CN, CN3x3,CSD,GLRLM,GLRLM3x3, HOG,LBP, LBP3x3)
	3. devset_topics.xml- XML File containing mapping of number to location name.
	4. devset/xml- Has xml files for each location which groups the user who has annotated the correspoding image in that particular location. (Useful for building tensors)
	5. devset/desccred- Contains user tagging credibility descriptors that give an automatic estimation of the quality of tag-image content relationships.
	6. devset/gt - Contains the ground truth data which consists of relevance ground truth (code rGT) and diversity ground truth (code dGT and dclusterGT).

Important: Please note that all the photos provided are under Creative Commons licenses . Each Flickr photo is provided with the license type and the owner’s name. 	
For more detailed information about the dataset, follow the link: http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/Div150Cred_readme.txt