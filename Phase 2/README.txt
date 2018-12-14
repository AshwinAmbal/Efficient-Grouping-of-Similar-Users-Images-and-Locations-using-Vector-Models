MWDB PROJECT - PHASE 2

Contributors: 
Ashwin Karthik Ambalavanan
Ajay Gunasekaran
Aditya Vikram Sharma
Balaji Gokulakrishnan 
Srivathsan Baskaran
Akbar Jamal Abbas

Student- Arizona State University

The codebase in this phase is used for performing dimensionality reduction on the existing Object-Feature space using different techniques and finding similarity between users, images and locations from features extracted from a set of images in flickr as given in the link: http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5

Check the detailed description of tasks in the pdf file attached with the project. The task code SHOULD be run in the order described for correct execution.

The "MWDB Project/Phase 2/Code" folder contains the entire project
The "MWDB Project/Phase 2/Outputs" folder contains document outputs for each individual task.
The "MWDB Project/Phase 2/Report" describes in detail what has been done in the project.

Description of items MWDB Project/Phase 1/Code Folder:
	1. Code/CSV- Contains required csv files for each task
	2. Code/Tasks- Contains the Task code for each task.
	3. Code/Sample_Inputs for Phase 2 - Has the example inputs given to each task.
	
Code/CSV/Task 1_2- Contains the CSV files TF_IDF_User, TF_IDF_Loc and TF_IDF_Img with user, location and image data categorized together respective in an RDBMS format.
Code/CSV/Task 3- Contains the color model files of all the locations written by "Task_3.py" after reduction using PCA, SVD or LDA as given by the user.
Code/CSV/Task 4_5- Contains the color model files of all the locations written by "Task_4.py" and "Task_5.py" after reduction using PCA, SVD or LDA as given by the 
					user.
Code/CSV/Task 4_5_Img- Contains the color model files of all the locations written by the "K_Means_Clustering.py" file which performs clustering on the set of images 
						and brings the number of images in a location down to 70 by taking number of clusters as 7 and taking 10 images from each cluster. These are the files read by "Task_4.py", "Task_5.py" and "Task_6.py"
					
Code/CSV/Task 6- Contains the color model files of all the locations written by "Task_6.py" after reduction using PCA, SVD or LDA as given by the user.
Code/CSV/Similarity_Matrix.csv- Written by "Task_6.py". It contains the location-location similarity matrix which is computed using all 10 color models for all the 
								locations in the devset. The task takes a long time to compute and hence it is better to store the results in this csv file to read back later when we need it. [Run this code only when dataset changes or if you do not already have the Similarity_Matrix.csv file]

The task code described below SHOULD be run in the order described for correct execution.

Code/Tasks/CSV_Writer_DB_User.py- The code is used to extract required data from the file devset_textTermsPerUser.txt and write the required 		
information in a database type format into a csv file. The column header in the written file would signify the user id's of the various users and the row headers would be annotations.

Code/Tasks/CSV_Writer_DB_Images.py- The code is used to extract required data from the file devset_textTermsPerImage.txt and write the required information in a database type format into a csv file. The column header in the written file would signify the user id's of the various users and the row headers would be annotations.

Code/Tasks/CSV_Writer_DB_Locations.py- The code below is used to extract required data from the file devset_textTermsPerPOI.txt and write the required information in a database type format into a csv file. The column header in the written file would signify the user id's of the various users and the row headers would be annotations.

Code/Tasks/Task_1_2.py- The code first reduces the terms to k-latent semantics using PCA, SVD or LDA as inputted by the user and the corresponding term-weight pairs are listed. The code then extracts 'k' similar users, images or locations (based on user input) for a given user, image or location id in the reduced space based on the tf-idf values.

Code/Tasks/Task_3.py- Given a visual descriptor model and a value 'k', the image feature space of the given color model for each location is reduced along the color model values and the corresponding term-weight pairs are listed. For example, for an image in CM model, there will be 9 values and given k = 5 those 9 values will be reduced to 5 values using PCA, SVD or LDA as inputted by the user. An Image ID is then inputted by the user for which the similarity needs to be computed. This image is compared with all other images in the dataset and the most similar 5 images and their corresponding locations are returned. To find 5 most similar locations to the location of the given image id, the average of similarity scores of the given image compared with all images of a particular location is calculated for each of the locations in the devset. The top 5 average scores and the corresponding location is returned.

Code/Tasks/K_Means_Clustering.py- The code below performs K-Means Clustering with a cluster size of 7 with 10 image samples taken from each cluster and written into the Task 4_5_Img folder. These images are used for Task_4, Task_5 and Task_6.

Code/Tasks/Task_4.py- Given a visual descriptor model and a value 'k', the image feature space of the given color model for each location is reduced along the color model values and the corresponding term-weight pairs are listed. The code is then used to extract 5 similar locations given a location id for the color model inputted previously. For each match, the code also lists the overall matching score.

Code/Tasks/Task_5.py- Given a value 'k', the image feature space of all color models for each location is reduced along the color model values and the corresponding term-weight pairs are listed. The code is then used to extract 5 similar locations given a location id. For each match, the code also lists the overall matching score cumulative of all the models. It also lists the individual contributions of the 10 color models.

Code/Tasks/Task_6.py- Location-Location Similarity matrix is formed which would be a symmetric matrix. This is essentially running "Task_5.py" for each location and finding the similarity score for that location with every other location. The resultant matrix is saved to "Similarity_Matrix.csv" file in the Code/CSV folder. SVD is then performed on this matrix and the top k-latent semantics is listed along with the corresponding term-weight pairs.

Code/Tasks/Task_7.py- The xml files in devset/xml gives a grouping of users, images, locations and corresponding tags common to all three. We use these values to construct a tensor with loc x user x images which would be a tensor of dimension 30 x 530 x 8912. This tensor would be very sparse. And as specified by the task, given a value 'k', we perform rank-k CP-Decomposition on this tensor and display the corresponding factor matrices. We then perform K-means clustering with value 'k' ('k' being the same inputted 'k' as before) on the three matrices to form k-groups of non-overlapping user, image and locations as required by the question.