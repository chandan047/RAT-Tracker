# RAT Tracker (Route-And-Track Tracker)

The base code is forked from [siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch). This code is modified as a part of project for the course "CSE586: Computer Vision 2", Pennsylvania State University, under the guidance of Prof. Robert Collins. 

The objective of this project is to study how Object Tracking can benefit from the properties of [Capusle Network](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf). We analyze the performance of Capsule Network on various motion classes and conclude some interesting results.

A clean PyTorch implementation of SiamFC tracker described in paper [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html). The code is evaluated on 7 tracking datasets ([OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS](http://ci2cv.net/nfs/index.html) and [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)), using the [GOT-10k toolkit](https://github.com/got-10k/toolkit).

## Model

![Model](https://github.com/chandan047/RAT-Tracker/blob/master/arch.png)


## Progress

- [x] Implement Capsule Network with dynamic routing between capsules [2]
- [x] Integrate with SiamFC [1]
- [x] Implement Adaptive routing for capsule networks[3]
- [x] Run RAT Tracker model on VOT 2018, OTB-100 datasets
- [x] Ablation study on motion-classes
- [ ] Re-train the model with loss on object classification/reconstruction
- [ ] Motion class group vs output layer of capsule network


## Performance (the scores are updated)

### GOT-10k

| Dataset | AO    | SR<sub>0.50</sub> | SR<sub>0.75</sub> |
|:------- |:-----:|:-----------------:|:-----------------:|
| GOT-10k | 0.334 | 0.352             | 0.099             |

The scores are comparable with state-of-the-art results on [GOT-10k leaderboard](http://got-10k.aitestunion.com/leaderboard).

### OTB / UAV123 / DTB70 / TColor128 / NfS

| Dataset       | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| OTB2015       | 0.584            | 0.788            |

### Comparison with [SiamFC](https://arxiv.org/abs/1606.09549)

|                       | SiamFC           | RAT Tracker      |
|:----------------------|:----------------:|:----------------:|
| Success Score (IOU)   | **58.97**            | 58.44            |
| Precision Score       | **79.20**            | 78.88            |
| Success Rate          | **74.33**            | 72.84            |
| Speed (fps)           | **66.90**            | 16.88            |

Performance of [SiamFC](https://arxiv.org/abs/1606.09549) is overall better than our RAT Tracker. However in the sub-class performance comparision figures below, we can see that our model outperforms SiamFC in certain categories of motion class.

Success Score (IOU)        |  Precision Score          |   Success Rate            |
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/chandan047/RAT-Tracker/blob/master/plots/success_score.png)  |  ![](https://github.com/chandan047/RAT-Tracker/blob/master/plots/precision_score.png)   |  ![](https://github.com/chandan047/RAT-Tracker/blob/master/plots/success_rate.png)

RAT Tracker outperforms **Illumination Variation (IV)**, **Background Clutters (BC)**, **Low Resolution (LR)** and **Motion Blur (MB)** videos. Performance on **Fast Motion (FM)** is competitive. There is a consistent gain in performance on these types of motion classes where object has similar orientation but the quality is low. Further analysis is required to understand this model.


## Installation

Install Anaconda, then install dependencies:

```bash
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv
# install GOT-10k toolkit
pip install got10k
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 9 popular tracking datasets.

## Training the tracker

1. Setup the training dataset in `tools/train.py`. Default is the GOT-10k dataset located at `~/data/GOT-10k`.

2. Run:

```
python tools/train.py
```

## Evaluate the tracker

1. Setup the tracking dataset in `tools/test.py`. Default is the OTB dataset located at `~/data/OTB`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/test.py
```

## Running the demo

1. Setup the sequence path in `tools/demo.py`. Default is `~/data/OTB/Crossing`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/demo.py
```

# References

1. [Fully Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549), Bertinetto et al.  
2. [Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf), Sabour et al.  
3. [Adaptive Routing Between Capsules](https://arxiv.org/abs/1911.08119), Ren et al.  
4. [OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)  
5. [VOT (2018)](http://votchallenge.net)  
