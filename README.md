# NYCU VRDL 2025 Spring HW1

StudentID: 111550132

Name: 張家睿

## Introduction
Use ResNeXt as model structure, with 22 different ramdom seed to train. Than use Model Soup/Greedy Soup/Model Stock to compose a stronger model.
#### Single model:
* training model: [code/ResNext_v4.py](code/ResNext_v4.py)
* testing model: [code/test_v4.py](code/test_v4.py)
* drawing confusion matrix: [code/confusion_matrix.py](code/confusion_matrix.py)
#### compose multi models to single model:
* model soup: [code/model_soup.py](code/model_soup.py)
* greedy soup: [code/greedy_soup.py](code/greedy_soup.py)
* model stock: [code/model_stock.py](code/model_stock.py)

## How to install
Download the [environment.yml](environment.yml), execute this lines in a computer with conda.
```
conda env create -f environment.yml -n env
# after creating env
conda activate env
python code/ResNext_v4.py
```

## Performance snapshot
![leaderboard.png](leaderboard.png)