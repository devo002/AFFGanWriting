[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# AFFGANwriting: A Handwriting Image Generation Method Based on Multi-feature Fusion
## Installation

```console
conda create --name AFFGanWriting python=3.7
conda activate AFFGanWriting
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/wh807088026/AFFGanWriting.git && cd AFFGanWriting
pip install -r requirements.txt

## Dataset preparation

The main experiments are run on [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) since it's a multi-writer dataset. Furthermore, when you have obtained a pretrained model on IAM, you could apply it on other datasets as evaluation, such as [GW](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database),  [RIMES](http://www.a2ialab.com/doku.php?id=rimes_database:start), [Esposalles](http://dag.cvc.uab.es/the-esposalles-database/) and
[CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/). 

## How to train it?

First download the IAM word level dataset, then execute `prepare_dataset.sh [folder of iamdb dataset]` to prepared the dataset for training.  
Afterwards, refer your folder in `load_data.py` (search `img_base`). 

Then run the training with:

```bash
./run_train_scratch.sh
```

**Note**: During the training process, two folders will be created: 
`imgs/` contains the intermediate results of one batch (you may like to check the details in function `write_image` from `modules_tro.py`), and `save_weights/` consists of saved weights ending with `.model`.

If you have already trained a model, you can use that model for further training by running:

```bash
./run_train_pretrain.sh [id]
```

In this case, `[id]` should be the id of the model in the `save_weights` directory, e.g. 1000 if you have a model named `contran-1000.model`.


## How to test it?

We provide two test scripts starting with `tt.`:

* `tt.test_single_writer.4_scenarios.py`: Please refer to Figure 4 of our paper to check the details. At the beginning of this code file, you need to open the comments in turns to run 4 scenarios experiments one by one.

* `tt.word_ladder.py`: Please refer to Figure 7 of our paper to check the details. It's fun:-P


## Citation

If you use the code for your research, please cite our paper:

```
To be updated...
```

### Implementation details
This work is partially based on the code released for [GANwriting](https://github.com/omni-us/research-GANwriting)
