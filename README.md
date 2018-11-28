### Taking the CNSE data set as an example, a two-channel GCN model is constructed to calculate the similarity between texts. 

### What's more, you can also change one channel to load the picture features and then this task is about cross modal retrevial. If you want to know more about the cross modal retrevial, you can read these papers.

- Yu J. et al. (2018) [Modeling Text with Graph Convolutional Network for Cross-Modal Information Retrieval.](https://link.springer.com/chapter/10.1007/978-3-030-00776-8_21) In: Hong R., Cheng WH., Yamasaki T., Wang M., Ngo CW. (eds) Advances in Multimedia Information Processing – PCM 2018. PCM 2018. Lecture Notes in Computer Science, vol 11164. Springer, Cham
- Yu J, Lu Y, Zhang W, et al. [Learning cross-modal correlations by exploring inter-word semantics and stacked co-attention[J].](https://doi.org/10.1016/j.patrec.2018.08.017) Pattern Recognition Letters, 2018.
- Lu Y., Yu J., Liu Y., Tan J., Guo L., Zhang W. (2018) [Fine-Grained Correlation Learning with Stacked Co-attention Networks for Cross-Modal Information Retrieval.](https://link.springer.com/chapter/10.1007/978-3-319-99365-2_19) In: Liu W., Giunchiglia F., Yang B. (eds) Knowledge Science, Engineering and Management. KSEM 2018. Lecture Notes in Computer Science, vol 11061. Springer, Cham

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







