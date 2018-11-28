简介: 以CNSE数据集为例，构建双通道GCN模型，计算文本间的相似度。

作者: 卢宇航 

##### train.py : 

- 参数准备，训练模型，

- main()：
  - 根据总词表数量，确定bow特征维度
  - 输入每篇文章得到特定的bow特征，保存成npy格式。

##### lib/utils.py : 预处理数据（如需修改输入数据、标签，从此入手）

- Build_graph（），通过邻接表构建图结构。
- load_wl（），读取词表。
- build_bow（），构建bow特征。
- prepair_data（），数据整体预处理。
- save_data（），保存处理好的数据，方便直接读取。
- load_data（），读取处理好的数据。

##### lib/model.py : 模型文件

##### /data/content_10_knn_graph.txt：以邻接表形式存在的图结构







