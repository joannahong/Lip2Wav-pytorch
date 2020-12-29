## Overview
This repository provides a PyTorch implementation of [Lip2Wav](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf).  

## Notice
I have tried to match official implementation as much as possible, but it may have some mistakes, so please be careful of using this implementation. Also, feel free to tell me any suggestion for this repository. Thank you!


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
You can download lrw datasets from the original [Lip2Wav repository](https://github.com/Rudrabha/Lip2Wav/tree/multispeaker), same format of dataset directories is used. [Grid dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/) is also trainable.
**Please preprocess first following steps listed in the original repository.**


## Training
* For training Lip2Wav-pyroch, run the following command.
```
python train_multi.py --data_dir=<dir/to/dataset> --log_dir=<dir/to/models>
```
* For training using pretrained model, run the following command.
```
python train_multi.py --data_dir=<dir/to/dataset> --log_dir=<dir/to/models> --ckpt_dir=<pth/to/pretrained/model>
```
  
## Inference
* For testing, run the following command.
```
python test.py --data_dir=<dir/to/dataset> --results_dir=<dir/to/save/results> --checkpoint=<pth/to/model>
```


## Pretrained Model
will be updated


## Results
will be updated



## Acknowledgements
This repository is modified from the original [Lip2Wav](https://github.com/Rudrabha/Lip2Wav) and highly based on [Tacotron2-Pytorch](https://github.com/BogiHsu/Tacotron2-PyTorch). I am so thankful for these great codes. 
