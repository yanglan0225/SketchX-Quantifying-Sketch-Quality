# Quantify Free-Hand Sketch Quality

This repo contains the codebase of a series of research projects focused on quantifying human free-hand sketch quality

- [Finding Badly Drawn Bunnies](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Finding_Badly_Drawn_Bunnies_CVPR_2022_paper.pdf) in ***CVPR2022***
- Not *That* Bunny: An Annotation-Free Approach for Human Sketch Quality Assessment ***Under Review***

## Environment settings
-- Pytorch 1.7.1 

## Structure of this repo
** dataloader.py ** - load [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset

** main.py ** -- main training and evaluation function

** models.py ** -- GACL implementations is here

** data_utils.py ** -- data processing functions

** config.py ** -- experiment settings parameters

## Trained models
coming soon

## Citation
If you use this code in your research, please kindly cite the following papers
```
@inproceedings{lancvpr,
  title={Finding Badly Drawn Bunnies},
  author={{Yang}, Lan and Pang, Kaiyue and Zhang, Honggang and Song, Yi-Zhe},
  booktitle={CVPR},
  year={2022},
}
```
