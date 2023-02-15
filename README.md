# medical_image_as_graph

### Create virtual enviornment 
```
conda env create -n graphenv -f environment.yml
conda activate graphenv
```


### To run an experiment execurte 
```
python main.py  --cnn_model_name {resnet18/efficienet-b0/densenet121} --gnn_model {GCN/GAT/GIN}  --use_saved_state yes   --superpixel_number {5/10/50/100/150/300}  --train {yes/no/True/False}
```
Example:
In this example we are testing on graph created on image with 5 superpixels, features extracted from resnet18 and GNN model used is GCN

```
python main.py  --cnn_model_name resnet18  --gnn_model GCN  --use_saved_state yes   --superpixel_number 5  --train no
```

To replicate the results, in colab use replicate_experiment.ipynb 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E2iWu7IsS2eK8jyZS1dD5ZK2cBsc6fly?usp=sharing)
