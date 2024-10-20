# Grid-6.0
Product Detection and Attribute Extraction using YOLOv8



A project aimed at detecting daily-use packed products and extracting relevant attributes (such as brand names, net weight, and MRP) from product packaging using YOLOv8.

📖 Table of Contents
Project Overview
Dataset Structure
Annotation Details
Training the Model
Text Extraction (OCR)
Results
Future Improvements
Project Structure
License
Contact
🚀 Project Overview
This project uses YOLOv8 for object detection and integrates Google Cloud Vision API for Optical Character Recognition (OCR) to extract information like:

Product Name
Brand
Net Weight
Expiration Date, etc.
With a dataset of over 2000+ images and 40+ different products, the goal is to train a robust model that can detect and extract product details from packaging.

🗂️ Dataset Structure
The dataset consists of product images and their corresponding label files with bounding boxes for the following attributes:

Product Name
Brand
Net Weight
Other Packaging Details
Directory Structure:
bash
Copy code
/Dataset
    /images
        /train          # Training images
        /val            # Validation images
    /labels
        /train          # Training labels (bounding boxes)
        /val            # Validation labels
📝 Annotation Details
The bounding box annotations follow the YOLOv8 format, which includes:

Class ID
Bounding Box Center (x, y)
Width and Height (normalized to image dimensions)
Example Annotation (YOLOv8 format):

plaintext
Copy code
<class_id> <x_center> <y_center> <width> <height>
Classes:
0: Kimia Dates
1: Kimia Dates Details
...
166: Red Label Net
🏋️‍♂️ Training the Model
We use YOLOv8 for training, leveraging its ability to handle object detection efficiently.

Steps to Train:
Install YOLOv8:

bash
Copy code
pip install ultralytics
Prepare your data.yaml file:

yaml
Copy code
names:
  0: Kimia Dates
  1: Kimia Dates Details
  ...
  166: Red Label Net

train: /path/to/images/train
val: /path/to/images/val
Train the Model:

python
Copy code
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Train
model.train(data='data.yaml', epochs=50, imgsz=640, project='Results', name='product_detection')
Inference:
Once the model is trained, use it to predict new images:

python
Copy code
results = model.predict(source='/path/to/test_image.jpg', save=True)
🔍 Text Extraction (OCR)
We use Google Cloud Vision API to extract text from detected bounding boxes for attributes like:

MRP
Brand
Net Weight
OCR Setup:
Install the Google Cloud Vision client:

bash
Copy code
pip install google-cloud-vision
Set up your credentials:

python
Copy code
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/credentials.json'
Text Extraction:

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
📊 Results
After training for 50 epochs, the model achieved:

mAP50: 98%
mAP50-95: 79.9%
The model successfully detects products and extracts key attributes from their packaging.

🔧 Future Improvements
Model Fine-Tuning: Continue improving the model with more data and experimenting with larger YOLOv8 variants.
Text Extraction Accuracy: Improve the OCR by enhancing preprocessing for better text clarity.
Support for Multi-Product Detection: Expand the model to detect and extract details from multiple products in a single image.
📂 Project Structure
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
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📧 Contact
For any inquiries or collaboration, feel free to reach out:

Name: Mohammed Arbaz
Email: mdarbaz3636@gmail.com
