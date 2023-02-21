# medical_image_as_graph

### Create virtual enviornment 
```
conda env create -n graphenv -f environment.yml
conda activate graphenv
```


### To run an experiment 
```
python main.py  --cnn_model_name {resnet18/efficienet-b0/densenet121} --gnn_model {GCN/GAT/GIN}  --use_saved_state yes   --superpixel_number {5/10/50/100/150/300}  --train {yes/no/True/False}
```
Example:
In this example we are testing on graph created on image with 5 superpixels, features extracted from resnet18 and GNN model used is GCN

```
python main.py  --cnn_model_name resnet18  --gnn_model GCN  --use_saved_state yes   --superpixel_number 5  --train no
```

To replicate the results in colab use **replicate_experiments.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E2iWu7IsS2eK8jyZS1dD5ZK2cBsc6fly?usp=sharing)

To replicate ensemlby result use **ensembling_result.ipynb**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aktjvqLi908s3VcKJVRDYa77ENQuo7AS?usp=sharing)

### To create new graph dataset from image, dowload chest_xray images and run the following commands
```
sh download.sh 

python main.py  --cnn_model_name {cnn_model_name}  --use_saved_state no  --superpixel_number {superpixel_value}
```
Example:
This will create a folder named chest_xray_graph and will create graph from images, using features of resnet18 and superpixel value of 5 i.e 5 node graph. 
```
python main.py  --cnn_model_name resnet18  --gnn_model GCN  --use_saved_state no  --superpixel_number 5
```

