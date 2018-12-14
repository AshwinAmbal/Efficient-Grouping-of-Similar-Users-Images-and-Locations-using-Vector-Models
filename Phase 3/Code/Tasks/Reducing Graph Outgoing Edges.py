import json

g = int(input("Enter the value of k (Graph to use): "))
column_sim_model = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(g)))

for k in range(1, g):
    for img in column_sim_model:
        column_sim_model[img] = column_sim_model[img][:g-k]
    json.dump(column_sim_model, open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_{}.json".format(g-k), 'w'))
    print("Done with ", g-k)
    
#column_sim_model = json.load(open("C:\\MWDB Project\\Phase 3\\Code\\CSV\\Img_Img_Graph_k_4.json"))

#column_sim_model
