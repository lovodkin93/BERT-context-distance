# BERT-context-distance


## 1. Install
Let's start with making sure you have the full repository, including all the git submodules.

`git clone --recursive https://github.com/lovodkin93/BERT-context-distance.git`

This will download the full repository.

Now, let's get your environment set up. Make sure you have conda installed, then run: 

```
cd BERT-context-distance/jiant
conda env create -f environment.yml
```
Now activate the `jiant` environment:

`conda activate jiant`

Now run:
```
pip install seaborn
pip install allennlp==0.8.4
pip install sacremoses
pip install pyhocon
```

## 2. Run
Go back to the `BERT-context-distance` directory and run the main function:
```
cd ..
python ./main.py
```
