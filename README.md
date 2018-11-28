Intro: Taking the CNSE data set as an example, a two-channel GCN model is constructed to calculate the similarity between texts.

Author:  Yuhang Lu

##### train.py : 

- Parameter preparation and model training

##### lib/utils.py : Pre-process data (if you need to modify the input data, labels, start from here)

- Build_graph（）, The graph structure is constructed by adjacency tables.
- load_wl（）, Load the vocabulary.
- build_bow（），Construct bagofwords features.
- prepair_data（）
- save_data（），Save data preprocessed.
- load_data（），Load data.

##### lib/model.py : 

##### /data/content_10_knn_graph.txt：Graph structure in the form of adjacency list







