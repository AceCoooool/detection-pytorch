# detection-pytorch
This repository is my reproduction of classical object detection in pytorch. （For own study and reference others' implementation --- the code quality may not good :flushed:）

![](png/demo.gif)

## Done

- [x] [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [x] [YOLO v2](https://arxiv.org/abs/1612.08242)
- [ ] [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)（without training）

Note：This implement is mainly based on the [amdegroot's ssd](https://github.com/amdegroot/ssd.pytorch) :+1:. 

**Detail of  instructions are in each sub-direcory**：[ssd](ssd/README.md)，[yolo](yolo/README.md)，[yolo3](yolo3/README.md)

## Pre-requisties

- Python 3.4+ 
- Pytorch 0.4 （Note：you can install through source or through follow conda）
  `conda install -c ostrokach-forge pytorch=0.4.0`
- OpenCV（optional，PIL.Image is also well）
- CUDA 8.0 or higher（optional）

## Dataset

PASCAL_VOC 07+12：follow the instructions in [amdegroot's ssd](https://github.com/amdegroot/ssd.pytorch)

1. **Download VOC2007 trainval & test**

   ```shell
   # specify a directory for dataset to be downloaded into, else default is ~/data/
   sh dataset/scripts/VOC2007.sh # <directory>
   ```

2. **Download VOC2012 trainval**

   ```shell
   # specify a directory for dataset to be downloaded into, else default is ~/data/
   sh dataset/scripts/VOC2012.sh # <directory>
   ```

（Note：if your dataset is not in `~/data`，please modify the `dataset/config.py` ’ s home parameter to you data path.）

## Performance

|  detection model  | mAP(07) | mAP(10) |                         Google Drive                         |                         Baidu Drive                          |
| :---------------: | :-----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    SSD (vgg16)    | 77.55%  | 80.10%  | [vgg_final.pth](https://drive.google.com/open?id=1D9pPJWEwK48DWf1mz18Wl4EZHrQCHw8n) | [vgg_final.pth](https://pan.baidu.com/s/1Hr4J4rbpKhyVpIPitPa3fg) |
|   SSD (res101)    | 75.97%  | 78.26%  | [resnet_final.pth](https://drive.google.com/open?id=10xIt9vbyibwjmifR-PHBJTQNYltrrNQA) | [resnet_final.pth](https://pan.baidu.com/s/1hZ9Tro840omWn-TKEndiVg) |
| YOLOv2 (official) | 73.40%  | 75.80%  | [yolo-voc.pth](https://drive.google.com/open?id=18J1jkENolbV_UW8l2Ds-RhNwFRUFtUZH) | [yolo-voc.pth](https://pan.baidu.com/s/1T8uD9SF8NlrSJKkZrCQDWg) |
|   YOLOv2 (here)   |         |         |                                                              |                                                              |
| YOLOv3(official)  |         |         | [yolo3.pth](https://drive.google.com/open?id=1c7LN6LraRE3ZxNl00ai_0xWtA74jMr7_) | [yolo3.pth](https://pan.baidu.com/s/1KkRrPjrz1CkMnuqPst64ig) |

Note：

1. The pretrained vgg model is converted from caffe and download from [amdegroot's ssd](https://github.com/amdegroot/ssd.pytorch)，and the pretrained res101 is coming from torchvision pretrained models.（I guess this is the reason why res101 based performance is worse than vgg based）
2. YOLOv2 official means the weights coming from the [pjreddie's website](https://pjreddie.com/darknet/yolo/)（can not find now :joy: ）
3. The data in ssd minus the mean and not divide 255. However, in the YOLO, the data without minus mean and divide 255. （No why，due to the pretrained basenet :sweat_smile:）

## Apology

There are sevaral important "functions" not  cantain in this repository：

- Only VOC dataset，not support other datasets (e.g. COCO dataset) 
- Only one card（GPU），not support multiprocess

（I am sorry for those. I only have one GPU card，and cannot finish the above functions～:neutral_face:）

## Reference

1. [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
2. [YAD2K](https://github.com/allanzelener/YAD2K)
3. [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)
4. [pytorch-yolo3](https://github.com/marvis/pytorch-yolo3)

Thanks for the great work by these authors.:heart: