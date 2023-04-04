# Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval
##### Author: Zhan Yang 
This is the offical PyTorch code for paper **"Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval"**, [Paper]().

## Please share a <font color='orange'>STAR ⭐</font> if this project does help

## Introduction
This is the novel and sophisticated PETL framework for the RS image-text retrieval task, the PyTorch source code of the paper "Parameter-Efficient Transfer Learning for Remote Sensing Image-Text Retrieval". Specifically, the proposed framework consists of the pretrained CLIP model, the multimodal remote sensing adapter (MRS-Adapter), and a hybrid multi-modal contrastive (HMMC) learning objective.

1. We design a simple yet effective loss function: the hybrid multi-modal contrastive (HMMC) loss for PETL-based RS image-text retrieval. Experimental results prove that the proposed HMMC loss is effective in further improving the performance on top of the proposed MRS-Adapter.
2. We provide comprehensive empirical studies for the PETL-based RS image-text retrieval task. Our qualitative and quantitative results demonstrate that the proposed method is promising and of great potential for practical applications.
3. Extensive experiments show that our approach can significantly reduce 98.9\% of fine-tuning parameters without performance sacrifice compared to full fine-tuning. Our retrieval performance exceeds traditional methods by 7-13\%. The comprehensive benchmark results are insightful for future research.



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


## Download Data
The RSITR dataset can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1F6WBQB-1PLqABh-uDv9m-KPdChakWcWY?usp=sharing). The download link is available below:
```
https://drive.google.com/drive/folders/1F6WBQB-1PLqABh-uDv9m-KPdChakWcWY?usp=sharing
```

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


## Remote Sensing Image-Text Retrieval
<p align="middle">
    <img src="fig/result.jpg">
</p>


## Reference
If you found this code useful, please cite the paper. Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then I will let you know when we update.
```
arxiv
```

## Acknowledgments
We benchmark extensive state-of-the-art PETL methods on the PE-RSITR task. We thank to the authors for releasing source codes. I would like to thank Xiong zhitong and Yuan yuan for helping the manuscript. I also thank the School of Artificial Intelligence, OPtics, and ElectroNics (iOPEN), Northwestern Polytechnical University for supporting this work.
