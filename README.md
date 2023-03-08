# DaTaKT

## Installation
Use the following command to install pyKT:

Create conda envirment.

```
conda create -n pykt python=3.6
source activate pykt
```


```
cd DaTaKT
pip install -e .
```

## Download Datasets & Preprocess

### Download
You can download datasets we used from [pyKT](https://pykt-toolkit.readthedocs.io/en/latest/datasets.html)

### Preprocess
```
cd examples
python data_preprocess.py --dataset_name=algebra2005
```

## Train & Evaluate
### Train
```
python wandb_bakt_time_train.py --use_wandb=0 --dataset_name=algebra2005
```
### Evaluate
```
python wandb_predict.py --use_wandb=0 --save_dir="/path/of/the/trained/model"
```
