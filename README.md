# Automated Real-Time Facial Emotion & Micro-Expression Classifier
A real-time computer vision application using OpenCV for facial detection, spatial ROI feature extraction, and real-time state classification.

📌 Project Overview
This repository contains a Computer Vision project demonstrating real-time face detection and region-of-interest (ROI) classification using OpenCV. The pipeline extracts visual features from dynamic webcam streams, applies grayscale transformations, identifies facial bounding coordinates, and classifies facial ROI states in real-time.

📂 Repository Structure
src/: Contains Python source code (emotion_detector.py).
Computer Vision Project Report.docx: Detailed report documenting the vision pipeline architecture and results.
README.md: Project documentation and quick-start guide (this file).

🛠 System Features & Functionality
This vision application was developed using Python and OpenCV:
- Real-Time Frame Ingestion: Captures video streams using OpenCV's VideoCapture interface.
- Spatial Feature Extraction: Employs Haar Feature Cascades for multi-scale face localization.
- ROI Preprocessing: Processes bounding region dimensions and converts color spaces for feature analysis.
- Visual Overlay: Draws dynamic bounding rectangles and status labels onto real-time display frames.

🚀 How to Run
1. Clone the repository:
   git clone https://github.com/Fireflie07/cv_project_24bac10013.git
   cd computer-vision-24bac10013

2. Install dependencies:
   pip install opencv-python numpy

3. Execute application:
   python src/emotion_detector.py

⚖️ License
This project is open-source and released under the MIT License.

👤 Author
Name: Elizabeth Maria George
Registration Number: 24BAC10013
