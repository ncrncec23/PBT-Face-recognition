# Real-Time Face Recognition Application Using Deep Learning

## Project Team

Name and Surname | E-mail address (FOI) | JMBAG | Github Username
------------  | ------------------- | ----- | ---------------------
Niko Crnčec | ncrncec23@student.foi.hr | 0016164582 | ncrncec23
Elena Pranjić | epranjic23@student.foi.hr | 0016164967 | epranjic23

## Domain Description
Real-time face recognition involves identifying faces in video streams or live feeds, which is useful in applications such as surveillance and security. Research in this field focuses on developing algorithms that can operate in real-world environments with varying lighting and camera angles. The project goal is to develop and implement deep learning-based techniques for real-time face recognition. Siamese networks are a specific deep neural network architecture well-suited for face recognition and verification tasks. They consist of two identical subnetworks sharing the same weights and learn to recognize the similarity between two input samples, such as two face images. Instead of classifying individual images, Siamese networks learn a comparison function that measures how similar or different two images are. This approach enables efficient recognition of new individuals even if the network has not seen their images during training, which is ideal for real-time face recognition systems with many users and varying conditions.

## Project Specification
**Project Goal** <br>
Develop a system for real-time face recognition using deep learning techniques and Siamese networks, capable of reliably identifying people in video streams despite changes in lighting, pose, and facial expressions.

**Input Data** <br>
<ul>
  <li>Video streams from a camera or prerecorded videos.</li>
  <li>Face images for training and model verification.</li>
</ul>

**Output Data**
<ul>
  <li>Identification of faces present in the video stream, including labels of recognized persons or indications of unknown faces.</li>
  <li>Real-time display of bounding boxes around detected faces and the person’s name.</li>
</ul>

**Application Features**
<ul>
  <li>Face detection in the video stream.</li>
  <li>Feature extraction using deep neural networks.</li>
  <li>Feature comparison using a Siamese network for recognition and verification.</li>
  <li>Management of the database of known faces (adding, removing, updating).</li>
  <li>Real-time display of results with visual indicators.</li>
</ul>

## Technologies and Equipment
The application is developed in Python. The trained model is available via the provided link. Due to local environment limitations, the model was trained in Google Colab, and the corresponding .ipynb notebook file is included in the GitHub repository. The project uses TensorFlow for model building and training, OpenCV for video processing, and the Kivy UI Framework for the user interface. The model was trained on the widely used face recognition dataset called Labelled Faces in the Wild (LFW), a standard in this application domain.

## References
1. _The application was developed based on this [video tutorial](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH), which provides an in-depth guide on deep learning, computer vision, Siamese model creation, and real-time person detection using OpenCV._
2. _In addition to the video, research on one-shot Siamese neural networks was used: [Siamese_Neural_Networks.pdf](https://github.com/user-attachments/files/20419437/Siamise_Neural_Networks.pdf)_



