# Digit classification model

## Overview

Classify digits.

## Folder structure

```
├── checkpoints
   └── model.pt
├── data
   └── test_data
├── data_loader
   └── data_loaders.py
├── models
   └── LeNet5.py
├── notebook
   └── notebook.ipynb
├── requirements
   └── requirements.yaml
├── utils
   └── utils.py
├── README.md
├── app.py
├── test.py
└── train.py
```

## Clone and install requirements

Linux or Mac OS
```
git clone https://github.com/ICE-opensource4/Digit_Classification_Model.git
conda env create -f requirements/requirements.yaml
```

## Dataset

Data: https://www.kaggle.com/competitions/digit-recognizer/data

License: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

Put this dataset in data folder.

## Train

```
python train.py [-h] [--seed SEED] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
```

## Test

Run train.py first and try it.
```
python test.py
```

## Demo

Run train.py first and get test data in data/test_data and try our deep learning model.
```
streamlit run app.py --server.fileWatcherType none
```


## License

```python
# MIT License
#
# Copyright (c) [year] [author]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
```