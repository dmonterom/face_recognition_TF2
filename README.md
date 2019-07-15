# Training a face Recognizer using ResNet50 + ArcFace in TensorFlow 2.0

The aim of this project is to train a state of art face recognizer using TensorFlow 2.0. The architecture chosen is a modified version of ResNet50 and the loss function used is [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), both originally developed by deepinsight in [mxnet](https://github.com/deepinsight/insightface). The choice of the network was made taking into account that it should have enough complexity to achive good results and that it should be relatively easy trainable from scratch. With regard to the loss function, it has been proved that ArcFace is the best loss function for face recognition to date.

The dataset used for training is the ~~CASIA-Webface~~ MS1M-ArcFace dataset used in [insightface](https://github.com/deepinsight/insightface), and it is available their [model zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The images are aligned using mtcnn and cropped to 112x112.

The results of the training are evaluated with lfw, cfp_ff, cfp_fp and age_db30, using the same metrics as deepinsight.

The full training and evaluation code is provided.

A Dockerfile is also provided with all prerequisites installed.


### Prerequisites

If you are not using the provided Dockerfile, you will need to install the following packages:

```
pip3 install tensorflow-gpu==2.0.0b1 pillow mxnet matplotlib==3.0.3 opencv-python==3.4.1.15 scikit-learn
```

### Preparing the dataset

Download the ~~CASIA-Webface~~ MS1M-ArcFace dataset from [insightface model zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) and unzip it to the dataset folder.

Convert the dataset to the tensorflow format:

```
cd dataset
mkdir converted_dataset
python3 convert_dataset.py
```

### Training the model

```
python3 train.py
```

The training process can be followed loading the generated log file (in output/logs) with tensorboard.

### Evaluating the model

The model can be evaluated using the lfw, cfp_ff, cfp_fp and age_db30 databases. The metrics are the same used in insightface.

Before launching the test, you may change the checkpoint path in the evaluation.py script.

```
python3 evaluation.py
```

### accuracy

The results are worse than the insightface model because: 
* ~~The training epochs are not enough.~~
* ~~The batch size should be bigger (16 used), as discused [here](https://github.com/deepinsight/insightface/issues/91) and [here](https://github.com/deepinsight/insightface/issues/86)~~
* ~~The training dataset used there has 85K ids and 5.8M images while the dataset used in this project has 10K ids and 0.5M images.~~ -> see model B

#### Trained models

##### model A
| model name    | train db| normalization layer |batch size| total_steps | download |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|
| model A | casia |batch normalization|16*8| 150k |[model a](https://drive.google.com/open?id=1RrVazZAWgDL26HxtacdeHfOADWERDUHK)|

| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9772|
| cfp_ff |0.9793|
| cfp_fp |0.8786|
| age_db30 |0.8752|


##### model B
| model name    | train db| normalization layer |batch size| total_steps | download |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|
| model B | ms1m |batch renormalization|16*8| 400k |[model b](https://drive.google.com/open?id=1PBDCw69nc3Ld02tj1n-ScFEbamzug7sW)|

| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9955|
| cfp_ff |0.9954|
| cfp_fp |0.9266|
| age_db30 |0.9540|

## TODO
* ~~The batch size must be bigger but the gpu is exhausted.~~ -> Now using batch ~~128~~ 96 by updating the gradients after several inferences. 
* ~~Further training of the net to improve accuracy.~~
* ~~Add batch renormalization for training using small batches.~~ ([link](https://arxiv.org/pdf/1702.03275.pdf))
* ~~Add group normalization for training using small batches.~~ ([link](https://arxiv.org/pdf/1803.08494.pdf))
* ~~Train the model with a bigger dataset.~~
* Add quantization awareness to training. This is not yet possible in TensorFlow 2.0 because it was part of the contrib module, which has been removed in the new version, as commented in [this issue](https://github.com/tensorflow/tensorflow/issues/27880).
* Test other network architectures.

## References
1. [InsightFace mxnet](https://github.com/deepinsight/insightface)
2. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
3. [InsightFace_TF](https://raw.githubusercontent.com/auroua/InsightFace_TF)
