# Grid-6.0
Product Detection and Attribute Extraction using YOLOv8
This project involves training a custom object detection model using YOLOv8 to detect daily-use packed products and extract relevant attributes such as brand names, net weight, and other details from the product packaging.

Project Overview
The goal of the project is to use computer vision techniques to:

Detect products from images of packaged goods.
Extract key attributes like product name, brand, net weight, MRP, and more.
Use YOLOv8 to train on a custom dataset of 40+ daily-use products with over 2000 annotated images.
Dataset
The dataset consists of:

Images: 2000+ images of daily-use products (e.g., Red Label tea, Johnson's baby soap, Kimia dates, etc.).
Labels: Corresponding annotations for each image, including bounding boxes for product detection and text extraction areas (brand name, net weight, etc.).
Dataset Structure
The dataset is structured as follows:

bash
Copy code
/Dataset
    /images
        /train          # Training images
        /val            # Validation images
    /labels
        /train          # Labels for training images
        /val            # Labels for validation images
Each image has a corresponding label file in the labels directory, with bounding box coordinates and class IDs.

Annotation
The products and their attributes are annotated using bounding boxes. Each product can have multiple attributes such as:

Product Name: e.g., Red Label
Brand Name: e.g., Brooke Bond
Net Weight: e.g., 1 kg
Variant: e.g., Garlic Mayonnaise, Dark Fantasy Choco Fils
Annotation Format (YOLOv8):
Each label file contains bounding box information and class IDs in the following format:

php
Copy code
<class_id> <x_center> <y_center> <width> <height>
Where the coordinates are normalized between 0 and 1.

Model
We use YOLOv8, a state-of-the-art object detection model, to detect products and their attributes from images.

YOLOv8 Configuration
The data.yaml file used to configure the training contains:

names: List of all the product names and attributes to be detected.
train: Path to the training images.
val: Path to the validation images.
nc: Number of classes (167 in this case).
data.yaml Example:
yaml
Copy code
names:
  0: Kimia Dates
  1: Kimia Dates Details
  2: Kimia Dates date
  ...
  166: red label net

train: /content/drive/MyDrive/mohammed_Arbaz/Dataset/images/train
val: /content/drive/MyDrive/mohammed_Arbaz/Dataset/images/val
nc: 167
Training the Model
The model is trained using the YOLOv8 framework. Here are the steps to train the model:

Installation
Install YOLOv8 by running:

bash
Copy code
pip install ultralytics
Install other necessary libraries:

bash
Copy code
pip install opencv-python pillow google-cloud-vision
Training Command
Run the following command to start training:

python
Copy code
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, depending on resources

# Train the model
model.train(
    data='/content/drive/MyDrive/mohammed_Arbaz/Dataset/data.yaml',  # Path to data.yaml
    epochs=50,          # Number of epochs
    imgsz=640,          # Image size
    project='/content/drive/MyDrive/mohammed_Arbaz/Results',  # Save results here
    name='product_detection'  # Experiment name
)
Model Results
After training, the best-performing weights will be saved in the /weights directory under the specified project. These weights can be used for inference or further fine-tuning.

Inference
To perform inference using the trained model:

python
Copy code
# Load the trained model
model = YOLO('/content/drive/MyDrive/mohammed_Arbaz/Results/product_detection/weights/best.pt')

# Run inference on a new image
results = model.predict(source='/path/to/test_image.jpg', save=True)
Text Extraction (OCR)
After detecting the product and bounding boxes for attributes like brand and net weight, you can use Google Cloud Vision API for Optical Character Recognition (OCR) to extract text from the images.

Google Cloud Vision Setup
Set up Google Cloud Vision API and provide credentials:

python
Copy code
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/credentials.json'
Extract text from detected bounding boxes:

python
Copy code
from google.cloud import vision

def detect_text(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response.text_annotations[0].description if response.text_annotations else None
Project Structure
bash
Copy code
/Dataset
    /images
        /train
        /val
    /labels
        /train
        /val
/Results                # Folder containing training results
    /weights            # Best and last weights from training
    /runs               # Training metrics and logs
data.yaml               # Dataset configuration file
train.txt               # Paths to training images
README.md               # Project documentation
Future Improvements
Fine-tuning the Model: Improve the accuracy by further fine-tuning the model with more annotated data or using a larger YOLOv8 variant.
Multi-Product Detection: Enhance the model to detect multiple products in one image.
Optimize OCR Performance: Integrate more advanced OCR techniques for accurate text extraction from the product labels.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or collaboration requests, please reach out to:

Name: Mohammed Arbaz
Email: mdarbaz3636@gmail.com
