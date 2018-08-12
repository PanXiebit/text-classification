# text-classification
Simplified models for text classification by tensorflow.

### TextCNN
- tensorflow implementation of [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)  
- more information of configuration in [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820).  

#### Requirements  
- tensorflow version-1.6.0  
- nltk version-3.3  
- wget version-3.2  
- tqdm version-4.24

#### Dataset
[DBpedia](https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz):The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. From each of thse 14 ontology classes, we randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000.

#### Training
The model was trained with NVidia 1080Ti, The model requires at least 12GB of GPU RAM. If your GPU RAM is smaller than 12GB, you can either decrease batch size.

```
python train.py --sentence_len 30 --batch_size 60
```

#### Test
```
python test.py --sentence_len 30 --batch_size 1000
```

#### Result
|version|train_acc|test_acc|
|---|---|---|
|v1|90.4%|87.8%|


- v1: sentence_len:30, filter_size:[3,4,5], batch_size:60, no batch_norm  
- v2: sentence_len:30, filter_size:[3,4,5],
batch_size:60, with batch_norm
