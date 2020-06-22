# Yolov5 Custom Data:

![](messigif.gif)

This is an implementation of the Ultralytics Yolov5 github repository on custom data. To run this code on the Football vs Cricket Data:

1. Clone this repo.
2. Install the dependenciers using

`$ pip install -U -r requirements.txt` 

3. Get the data from [kaggle](https://www.kaggle.com/mlwhiz/detection-footballvscricketball) and put it into the training folder

4. Run:

```
# Train yolov5l on custom dataset for 300 epochs
$ python train.py --img 640 --batch 16 --epochs 300--data training/dataset.yaml --cfg training/yolov5l.yaml --weights ''
```

5. To Predict:
`python detect.py --weights weights/best.pt`


You can read more on my blog : [Create an End to End Object Detection Pipeline using Yolov5](https://lionbridge.ai/articles/create-an-end-to-end-object-detection-pipeline-using-yolov5/)