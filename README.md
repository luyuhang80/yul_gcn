### Taking the CNSE data set as an example, a two-channel GCN model is constructed to calculate the similarity between texts. 

### What's more, you can also change one channel to load the picture features and then this task is about cross modal retrevial.

- Lu Y., Yu J., Liu Y., Tan J., Guo L., Zhang W. (2018) [Fine-Grained Correlation Learning with Stacked Co-attention Networks for Cross-Modal Information Retrieval.](http:https://link.springer.com/chapter/10.1007/978-3-319-99365-2_19) In: Liu W., Giunchiglia F., Yang B. (eds) Knowledge Science, Engineering and Management. KSEM 2018. Lecture Notes in Computer Science, vol 11061. Springer, Cham

----



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

##### data/content_10_knn_graph.txt：Graph structure in the form of adjacency list







