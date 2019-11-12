# Dependency Forest for Medical Relation Extraction
Code corresponding to our paper "Leveraging Dependency Forest for Neural Medical Relation Extraction" at EMNLP 2019

## Introduction

Folder "biaffine_forest" is a [deep biaffine parser](https://arxiv.org/abs/1611.01734) that supports producing dependency forests as introduced in our paper. It is obtained by adapting the original [dozat-parser](https://github.com/tdozat/Parser-v1).
To generate forests, use "--nbest" or "--cubesparse" *instead of* the traditional "--test" option when decoding.
"--nbest" and "--cubesparse" correspond to our "KbestEisner" and "Edgewise" method, respectively.
Parser training remains the same as the original system.
As mentioned by the original system description, this parser *only* works on Python 2 and TF 0.1.2, a very old version.

Folder "re_forest_grn" is our main relation extraction (RE) system based on graph recurrent network (GRN) for consuming dependency trees/forests.
There are several scripts within the "re_forest_grn/data" folder to help generating necessary data, such as word embeddings.
Training and decoding shells are also provided.
This model is based on Python 2 and TF 1.8.0.

## Cite 

If our work helps your research/system, please cite our work with the following bibtex file. 

```
@inproceedings{song2019leveraging,
  title={Leveraging Dependency Forest for Neural Medical Relation Extraction},
  author={Song, Linfeng and Zhang, Yue and Gildea, Daniel and Yu, Mo and Wang, Zhiguo and others},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={208--218},
  year={2019}
}
```
