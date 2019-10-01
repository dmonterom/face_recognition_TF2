# Training a face Recognizer using ResNet50 + ArcFace in TensorFlow 2.0

The aim of this project is to train a state of art face recognizer using TensorFlow 2.0. The architecture chosen is a modified version of ResNet50 and the loss function used is [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), both originally developed by deepinsight in [mxnet](https://github.com/deepinsight/insightface).

The dataset used for training is the ~~CASIA-Webface~~ MS1M-ArcFace dataset used in [insightface](https://github.com/deepinsight/insightface), and it is available their [model zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The images are aligned using mtcnn and cropped to 112x112.

The results of the training are evaluated with lfw, cfp_ff, cfp_fp and age_db30, using the same metrics as deepinsight.

The full training and evaluation code is provided.

A Dockerfile is also provided with all prerequisites installed.

**UPDATE** Increased single GPU batch size by updating the gradients after several inferences (96 in TESLA P100).

**UPDATE** Added Batch Renormalization and Group Normalization for training using smaller batches.

**UPDATE** Added regularization loss decay coefficient for reducing the impact of the regularization loss when the inference loss becomes smaller.

**UPDATE** Added multi gpu training code. It uses the experimental central storage strategy, which stores all the variables in the CPU and allows increasing the batch size on each GPU (128 for each TESLA P100).

**UPDATE** Added model C, trained with 3 gpus TESLA P100.

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

For training using 1 GPU:

```
python3 train.py
```

For training using multiple GPU:

```
python3 train_multigpu.py
```

The training process can be followed loading the generated log file (in output/logs) with tensorboard.

### Evaluating the model

The model can be evaluated using the lfw, cfp_ff, cfp_fp and age_db30 databases. The metrics are the same used in insightface.

Before launching the test, you may change the checkpoint path in the evaluation.py script.

```
python3 evaluation.py
```

### Trained models

##### model A
| model name    | train db| normalization layer |reg loss|batch size|gpus| total_steps | download |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model A | casia |batch normalization|uncontrolled|16*8|1| 150k |[model a](https://drive.google.com/open?id=1RrVazZAWgDL26HxtacdeHfOADWERDUHK)|

| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9772|
| cfp_ff |0.9793|
| cfp_fp |0.8786|
| age_db30 |0.8752|


##### model B
| model name    | train db| normalization layer |reg loss|batch size|gpus| total_steps | download |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model B | ms1m |batch renormalization|uncontrolled|16*8|1| 768k |[model b](https://drive.google.com/open?id=1PBDCw69nc3Ld02tj1n-ScFEbamzug7sW)|

| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9962|
| cfp_ff |0.9964|
| cfp_fp |0.9329|
| age_db30 |0.9547|

##### model C (multigpu)
| model name    | train db| normalization layer |reg loss|batch size|gpus| download |
| ----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| model C | ms1m |batch renormalization|uncontrolled|384|3|[model c](https://drive.google.com/open?id=1VqxJVsTARgRNACsscgPPQt7UJxalLlnz)|

| dbname | accuracy |
| ----- |:-----:|
| lfw |0.9967|
| cfp_ff |0.9970|
| cfp_fp |0.9403|
| age_db30 |0.9652|

## TODO
* ~~The batch size must be bigger but the gpu is exhausted. -> Now using batch ~~128~~ 96 by updating the gradients after several inferences.~~ -> Now using 2 GPU with batch size 128 on each GPU with the central storage strategy.
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
4. [Batch Renormalization](https://arxiv.org/pdf/1702.03275.pdf)
5. [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)
6. [Multi GPU Central Storage Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/CentralStorageStrategy)
