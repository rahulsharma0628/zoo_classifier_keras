# Zoo Classifier
Classify animals such as dogs, cats and pandas in the images using convolution neural networks. Keras is an easy-to-use library to build and evaluate neural network models. I have used an NVIDIA GeForce MX150 to train the model.

# Requirements
* tensorflow-gpu
* keras
* cv2
* argparse
* numpy
* os
* matplotlib

# Instructions

* Please clone this repository on your local system and create `img` folder insider `data` folder
* Download and unzip the [images.zip](https://psu.box.com/s/33kmt38s1p22bs5ee6l3mq21mj0ivnus) into `img` folder
* To train the model, go to the main directory and run:<br> `$ python train_model.py --dataset data/img --model <model_name>.model --plot <loss_plot_name>.png`
* To classify any image, use the sample images in data/test_image folder or you can upload your in the same folder and run the following code: <br>
  `$ python predict_model.py --image data/test_image/<image_name>.jpg --model <model_name>.model`
  <br>
  
For any queries, please reach out to me at **rahulsharma0628@gmail.com**
