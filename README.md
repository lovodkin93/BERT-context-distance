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

## Suggested Citation
if you use this repository for academic research, please cite the following citation:
```
@inproceedings{slobodkin-etal-2021-mediators,
    title = "Mediators in Determining what Processing {BERT} Performs First",
    author = "Slobodkin, Aviv  and
      Choshen, Leshem  and
      Abend, Omri",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.8",
    doi = "10.18653/v1/2021.naacl-main.8",
    pages = "86--93",
    abstract = "Probing neural models for the ability to perform downstream tasks using their activation patterns is often used to localize what parts of the network specialize in performing what tasks. However, little work addressed potential mediating factors in such comparisons. As a test-case mediating factor, we consider the prediction{'}s context length, namely the length of the span whose processing is minimally required to perform the prediction. We show that not controlling for context length may lead to contradictory conclusions as to the localization patterns of the network, depending on the distribution of the probing dataset. Indeed, when probing BERT with seven tasks, we find that it is possible to get 196 different rankings between them when manipulating the distribution of context lengths in the probing dataset. We conclude by presenting best practices for conducting such comparisons in the future.",
}

```
