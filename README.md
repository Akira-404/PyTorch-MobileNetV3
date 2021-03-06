# PyTorch-MobileNetV3
基于PyTorch的MobileNetV3的实现

**测试环境：Ubuntu-20.04，1050Ti**  

## 文件说明  

**model**.py:MobileNetV3模型  

**feature_map**.txt:网络结构  

**analyze_feature_map**.py:运行生成feature_map.txt  

**train_net**.py:训练脚本  

**test_net**.py:测试脚本    

**class_indices**.json:自动生成的目标索引文件  

**data_perparation**.py:用于分割训练集和测试集  

**my_dataset**.py:继承DataSet类，用于实现DataLoader  

**.pth**:权重文件  