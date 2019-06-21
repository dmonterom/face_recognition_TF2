Download dataset from:

https://www.dropbox.com/s/lfluom5ybqqln02/faces_CASIA_112x112.zip?dl=0

The images in the dataset are already filtered and aligned using mtcnn.

Convert the dataset to tensorflow format:

cd dataset
python3 convert_dataset.py

# Project Title

Training a face Recognizer using ResNet50 + ArcFace in TensorFlow 2.0

## Getting Started

The aim of this project is to train an state of art face recognizer using TensorFlow 2.0. The architecture chosen is ResNet50 + ArcFace, also known as LResNet50E-IR, developed by deepinsight in mxnet (https://github.com/deepinsight/insightface).

The dataset used for training is the CASIA-Webface dataset used by deepinsight, and can be downloaded from their [model zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). The images are aligned using mtcnn and cropped to 112x112.

The results of the training are evaluated with lfw, using the same metrics as deepinsight.

A Dockerfile is also provided with all prerequisites installed.

The full training and testing code is provided.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
