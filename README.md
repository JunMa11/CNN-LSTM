# CNN-LSTM

2.5D classification model for 3D medical imaging

## Installation
1. create env and clone the repo
```bash
conda create -n cnn_lstm python==3.10
conda activate cnn_lstm
git clone https://github.com/JunMa11/CNN-LSTM.git
```

2. Install the packages:
```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Training

write your custom dataset, define your preprocessing process and data augmentations.
For CT you can use train.py as reference
For MRI you can use train_hcm as reference