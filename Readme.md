# Brain Tumor Detection Using Deep Learning

Brain Tumor Detection with VGG19 and Gradio in Google Colab
This project leverages the power of deep learning to classify MRI brain images into tumorous or non-tumorous categories. Utilizing the VGG19 architecture for feature extraction and TensorFlow for model training, we aim to achieve high accuracy in detecting brain tumors. The project also includes a Gradio interface for easy model deployment and interaction, enabling users to upload MRI images and receive predictions directly through a web app.

## Project Overview

Brain tumors are a significant health challenge worldwide, with early diagnosis being crucial for effective treatment outcomes. This project applies Convolutional Neural Networks (CNNs), specifically using a pre-trained VGG19 model and TensorFlow/Keras, to distinguish between tumorous and non-tumorous MRI images of the brain.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed:
- Python 3.6+
- TensorFlow 2.x
- Keras
- Gradio
- Numpy
- Matplotlib

You can install the required libraries using `pip`:

```bash
pip install tensorflow keras gradio numpy matplotlib



Getting Started in Google Colab
Step 1: Setup Environment
Ensure you're using a Python environment with Google Colab. The first step involves installing necessary Python packages.

bash
Copy code
!pip install tensorflow keras gradio numpy matplotlib
Step 2: Import Libraries
Import all required libraries to build and train the model.

python
Copy code
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image
Step 3: Model Building
Load VGG19 as the base model, add custom layers on top, and compile the model.

python
Copy code
vgg19_base = VGG19(include_top=False, weights='imagenet', input_shape=(240, 240, 3))
vgg19_base.trainable = False

x = Flatten()(vgg19_base.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=vgg19_base.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
Step 4: Prepare Your Data
Organize your MRI data into training and validation directories, each with 'tumorous' and 'non-tumorous' subdirectories. Use ImageDataGenerator to load and augment images.

Step 5: Model Training
Train your model using model.fit(), passing it your data generators for training and validation.

Step 6: Deploy with Gradio
After training, deploy your model with a Gradio interface for easy interaction.

python
Copy code
def predict_mri(img):
    img = img.resize((240, 240))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Tumorous (Probability: {:.2f}%)".format(prediction[0][0] * 100)
    else:
        return "Not Tumorous (Probability: {:.2f}%)".format((1 - prediction[0][0]) * 100)

iface = gr.Interface(fn=predict_mri, inputs=gr.inputs.Image(), outputs="text", title="MRI Tumor Detector")
iface.launch(share=True)
Contribution
Contributions to the project are welcome. Please ensure to follow best practices for code contributions.

Fork the repository.
Create a new feature branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a Pull Request.




License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The VGG19 model, pre-trained on ImageNet, for serving as an effective feature extractor in this project.
TensorFlow and Keras for providing the tools to build and train deep learning models.
Gradio for enabling the easy creation of web interfaces to interact with deep learning models.
The MRI dataset obtained from Kaggle, which has been instrumental in training and evaluating our model. The dataset comprises MRI images categorized into tumorous and non-tumorous classes, providing a valuable resource for developing and testing our predictive model.
This project aims to contribute to the ongoing efforts in medical imaging and diagnosis through deep learning, offering a tool that could potentially aid in the early detection of brain tumors