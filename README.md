# Face_Detection_Theta
Implementing Face Detection algorithms to accomplish @home tasks

The algorithm finds faces on the image using OpenCV's implementation of Viola-Jones algorithms.
Viola-Jones algorithm consists on finding features of a face using Haar Cascades and applying a Adaboost to speed up the process.

The neural net to classify age and gender were taken from:

https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/

The dataset (wiki dataset) to test the algorithm for brahur/brasero 2020 demo can be found on:

https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

In the face_demo_brasero.py file, the script filters the images that do NOT have a face on them. The scripts plots at the left the
algorithms prediction and at the right the ground truth. 
 
