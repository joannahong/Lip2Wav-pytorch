## Overview
This repository provides a PyTorch implementation of [Lip2Wav](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf). This project is highly based on 

## Notice
I have tried to match official implementation as much as possible, but it may have some mistakes, so please be careful of using this implementation. Also, feel free to tell me any suggestion for this repository. *Sample results will be released soon.* Thank you!


## Requirements
* python >= 3.5.2
* torch >= 1.0.0
* numpy
* scipy
* pillow
* inflect
* librosa
* Unidecode
* matplotlib
* tensorboardX
* ffmpeg  <code>sudo apt-get install ffmpeg</code>


## Datasets
You can download datasets from the original [Lip2Wav repository](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pd), same format of dataset directories is used. **Please preprocess first following steps listed in the original repository.**


## Training
* For training Lip2Wav-pyroch, run the following command.
```
python train.py --data_dir=<dir/to/dataset> --log_dir=<dir/to/models>
```
* For training using pretrained model, run the following command.
```
python train.py --data_dir=<dir/to/dataset> --log_dir=<dir/to/models> --ckpt_dir=<pth/to/pretrained/model>
```
  
## Inference
* For testing, run the following command.
```
python test.py --data_dir=<dir/to/dataset> --results_dir=<dir/to/save/results> --checkpoint=<pth/to/model>
```


## Pretrained Model
Pretrained model of Chemistry Lectures is only available now. You can download the model [here](https://www.dropbox.com/sh/p6ljz9knhegxudl/AAAe63m0mpOZjUDbXbmkKROla?dl=0).


## Acknowledgements
This repository is modified from the original [Lip2Wav](https://github.com/Rudrabha/Lip2Wav) and [Tacotron2-Pytorch](https://github.com/BogiHsu/Tacotron2-PyTorch). I am so thankful for these great codes. 
