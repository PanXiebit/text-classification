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

### fasttext
tensorflow implementation of "[Bag of Tricks for Efficient Text Classification, Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov](https://arxiv.org/pdf/1607.01759.pdf)"

详细见：[文本分类系列1-fasttext](http://www.panxiaoxie.cn/2018/05/23/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E7%B3%BB%E5%88%971-fasttext/)



详细见：[文本分类系列2-textCNN](http://www.panxiaoxie.cn/2018/05/30/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E7%B3%BB%E5%88%972-textCNN/)

### TextRCNN
tensorflow implementation of "[Recurrent Convolutional Neural Networks for Text Classification](http://scholar.google.com/scholar?q=Recurrent+Convolutional+Neural+Networks+for+Text+Classification&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart)"

详细见：[文本分类系列4-textRCNN](http://www.panxiaoxie.cn/2018/06/01/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E7%B3%BB%E5%88%974-textRCNN/)


### Hierarchical Attention Networks
tensorflow implementation of [hierarchical attention networks for document classification](http://scholar.google.com/scholar?q=hierarchical+attention+networks+for+document+classification+github&hl=zh-CN&as_sdt=0&as_vis=1&oi=scholart)

详细见：[文本分类系列5-Hierarchical Attention Networks](http://www.panxiaoxie.cn/2018/06/03/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E7%B3%BB%E5%88%975-Hierarchical-Attention-Networks/#more)

### End-to-End Memory Networks
tensorflow implementation of [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895)

详细见：[QA1 memory networks 论文笔记](http://www.panxiaoxie.cn/2018/06/10/QA1-memory-networks-%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/#more)

And in the blog, there are more notes about  memory networks:
- Memory Networks
- End to End memory networks
- Dynamic memory networks
- DMN+


### reference

- [brightmart/text_classification](https://github.com/brightmart/text_classification)
- [roomylee/rcnn-text-classification](https://github.com/roomylee/rcnn-text-classification)
