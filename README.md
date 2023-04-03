# PE-RSITR: Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval
##### Author: Zhan Yang 
This is the offical PyTorch code for paper **"PE-RSITR: Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval"**, [Paper]().

## Please share a <font color='orange'>STAR ⭐</font> if this project does help

## Introduction
This is Multi-Granularity Visual Language Fusion (MGVLF) Network, the PyTorch source code of the paper "PE-RSITR: Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval". 
It is built on top of the [TransVG](https://github.com/djiajunustc/TransVG) in PyTorch. 
Our method is a transformer-based method for visual grounding for remote sensing data (RSVG). 
It has achieved the SOTA performance in the RSVG task on our constructed RSVG dataset.


### Network Architecture
<p align="middle">
    <img src="fig/model.jpg">
</p>



## Requirements and Installation
We recommended the following dependencies.
- Python 3.10.8
- PyTorch 1.13.1
- NumPy 1.23.4
- cuda 11.6
- opencv 4.7.0
- torchvision 0.14.1

## File structure
We expect the directory and file structure to be the following:
```
./                      # current (project) directory
├── layers/             # Files for implementation of PE-RSITR model
├── checkpoint/         # Savepath of pth/ckpt and pre-trained model
├── logs/               # Savepath of logs
├── vocab/              # 
├── data.py             # Load data
├── engine.py           # Functions of training, validation, and test
├── loss.py             # Implementation of loss function
├── utils.py            # Some scripts for data processing and helper functions 
├── vocab.py            # 
├── train.py            # Main code for training, validation
├── test.py             # Main code for test
├── README.md
└── data/                        # Dataset
    ├── rsicd_precomp/           # RSICD
        ├── rsicd_images/        # Remote sensing images
        ├── train_caps.txt       # Captions of training and validation set
        ├── train_filename.txt   # Image name of training and validation set
        ├── test_caps.txt        # Captions of test set
        └── test_filename.txt    # Image name of test set
    ├── rsitmd_precomp/          # RSITMD
        ├── rsitmd_images/       # Remote sensing images
        ├── train_caps.txt       # Captions of training and validation set
        ├── train_filename.txt   # Image name of training and validation set
        ├── test_caps.txt        # Captions of test set
        └── test_filename.txt    # Image name of test set
    ├── ucm_precomp/             # UCM
        ├── ucm_images/          # Remote sensing images
        ├── train_caps.txt       # Captions of training and validation set
        ├── train_filename.txt   # Image name of training and validation set
        ├── test_caps.txt        # Captions of test set
        └── test_filename.txt    # Image name of test set
```

## Training and Evaluation
```
python train.py
```

Run ```train.py``` to train the models.
Evaluate trained models using ```test.py```.

## Reference
If you found this code useful, please cite the paper. Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then I will let you know when we update.
```
arxiv
```

## Acknowledgments
Our DIOR-RSVG is constructed based on the [DIOR](http://www.escience.cn/people/JunweiHan/DIOR.html) remote sensing image dataset. 
We thank to the authors for releasing the dataset. Part of our code is borrowed from [TransVG](https://github.com/djiajunustc/TransVG). 
We thank to the authors for releasing codes. I would like to thank Xiong zhitong and Yuan yuan for helping the manuscript. 
I also thank the School of Artificial Intelligence, OPtics, and ElectroNics (iOPEN), Northwestern Polytechnical University for supporting this work.
