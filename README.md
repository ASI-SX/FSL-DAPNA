# [Few-Shot Learning as Domain Adaptation: Algorithm and Analysis](https://arxiv.org/pdf/2002.02050.pdf)
Jiechao Guan, Zhiwu Lu, Tao Xiang, Ji-Rong Wen

# Abstract
To recognize the unseen classes with only few samples, few-shot learning (FSL) uses prior knowledge learned from the seen classes. A major challenge for FSL is that the distribution of the unseen classes is different from that of those seen, resulting in poor generalization even when a model is meta-trained on the seen classes. This class-difference-caused distribution shift can be considered as a special case of domain shift. In this paper, for the first time, we propose a domain adaptation prototypical network with attention (DAPNA) to explicitly tackle such a domain shift problem in a meta-learning framework. Specifically, armed with a set transformer based attention module, we construct each episode with two sub-episodes without class overlap on the seen classes to simulate the domain shift between the seen and unseen classes. To align the feature distributions of the two sub-episodes with limited training samples, a feature transfer network is employed together with a margin disparity discrepancy (MDD) loss. Importantly, theoretical analysis is provided to give the learning bound of our DAPNA. Extensive experiments show that our DAPNA outperforms the state-of-the-art FSL alternatives, often by significant margins.

# Citation
If you find it useful, please consider citing our work using the bibtex:

    @misc{guan2020fewshot,
      title={Few-Shot Learning as Domain Adaptation: Algorithm and Analysis},
      author={Jiechao Guan and Zhiwu Lu and Tao Xiang and Ji-Rong Wen},
      year={2020},
      eprint={2002.02050},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

# Environment
* Python 3.7
* Pytorch 1.3.1

# Get Started 
## Data Preparation
  1. Folder '\data' should contain the raw images of 3 FSL datasets (e.g. miniImageNet, tieredImageNet, CUB). We download the original images of [ImageNet](http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz) to construct mini- and tieredImageNet datasets based on the [splitting strategies](https://github.com/JiechaoGuan/FSL-DAPNA/tree/master/filelists). We download the [CUB](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) dataset and use the bounding-box images. You can download the [miniImageNet](https://pan.baidu.com/s/1vnv7z_FAlginsucRKg-wng) (Enter Code: 4u9g, 2.86 Gb) and [CUB](https://pan.baidu.com/s/1KVqQYWmmBRAyd2-F9HUW-g) (Enter Code: kscu, 2.23 Gb) processed by us. The tieredImageNet zip exceeds 88 Gb so we are unable to upload it to Baidu Disk. Email to me (`guanjiechao0660@ruc.edu.cn`) if you want it.
  
  2. Folder '\saves' should include the pretrained WRN-28-10 models on three FSL datasets. You can pretrain a new one by following below instructions, or use our pretrained models: [miniImageNet](https://pan.baidu.com/s/1Li_VD4lH5u2oIwgvTw9eqg) (Enter Code: 2p7s), [tieredImageNet](https://pan.baidu.com/s/1KX_AH7xWsLbQQd4FninPhA) (Enter Code: zigs), [CUB](https://pan.baidu.com/s/1ipqChPjY3TMfJbae7lLO0w) (Enter Code: 1h2y).
## Model Training and Test
    --Standard FSL Setting
    1. Pre-train and save a model.
      -- python pretrain.py
    2. Train a DAPNA model.
      -- sh train_proto_mdd.sh
    3. Evaluate DAPNA's performance.
      -- python eval.py

    --Cross domain FSL setting
    1. Train a DAPNA model.
      -- sh cross_domain_train_proto_mdd.sh
    2. Evaluate DAPNA's performance.
      -- python cross_domain_eval.py

# Reference
We thank following repos providing helpful components/functions in our work.
1. A Closer Look at Few-shot Classification https://github.com/wyharveychen/CloserLookFewShot
2. Learning Embedding Adaptation for Few-Shot Learning https://github.com/Sha-Lab/FEAT
3. Bridging Theory and Algorithm for Domain Adaptation https://github.com/thuml/MDD
