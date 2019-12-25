
This repo is part of the object detection post:

To run the code in this repo: 

1. Download/Clone this repo

```
git clone https://github.com/MLWhiz/object_detection
```

2. Clone Mask_RCNN repo from matterport in the same object_detection folder:

```
git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN
pip install -r requirements.txt
```

3. Copy the guns_and_swords directory to the Mask_RCNN/samples directory

```
cp -r guns_and_swords/* Mask_RCNN/samples
```

4. go to samples directory and run

```
cd Mask_RCNN/samples
# To Train a new model starting from pre-trained COCO weights
 python3 gns.py train - dataset=/path/to/dataset - weights=coco
# To Resume training a model that you had trained earlier
 python3 gns.py train - dataset=/path/to/dataset - weights=last
# To Train a new model starting from ImageNet weights
 python3 gns.py train - dataset=/path/to/dataset - weights=imagenet
```

5. Use the notebooks in this directory to visualize dataset and do predictions.