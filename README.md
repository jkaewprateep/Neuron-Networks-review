# Neuron-Networks-review
Review of neuron networks class material from IBM data science

## Jupiter notebook files ##
#### Images processing ####
##### 1.0_load_and_display_data.ipynb ##### 
- Initial dataset from simple images using IO, PIL, matplotlib and pandas.
##### 2.1_data_loader_PyTorch.ipynb ##### 
- Prepare the dataset by creating folders for each type of target categorized for AI-supervised learning with sample sizes and ratios.
##### 2.1.2_Images_with_python_library_CV.ipynb ##### 
- Images concatenation, file string attribute manipulation, important image property size, identities, and dimension, image plotting and negative image, image label system conversion using CV2, save and read images using CV2, colour label conversion grey scales, colour scales, negative colurs, RGBA, RGB, *RGBY system. Image plot and sub-images plot, indexing and image cropping, image array values copy for parallel process images system, image colour channels and CV2 image colour system conversion sample BGR_to_RGB.  
##### 2.2.1_basic_image_manipulation_PIL.ipynb ##### 
- Introduce Python array management library Numpy, copy image array for new identity of same value for parallel images processing, object Id value and references, working with images as an array, image array properties, using Numpy array function to working with image sample flip and ImageOps from PIL, image array cropping using Numpy and Python array fundamental and value update, PIL image draw with fill colour function, * image font from PIL ImageFont, image overlay with cropped image.

##### PIL ImageOps rotation for image.transpose() function #####
```
flip = {
         "FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
         "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
         "ROTATE_90": Image.ROTATE_90,
         "ROTATE_180": Image.ROTATE_180,
         "ROTATE_270": Image.ROTATE_270,
         "TRANSPOSE": Image.TRANSPOSE, 
         "TRANSVERSE": Image.TRANSVERSE
        }
```

##### 2.2.2_basic_image_manipulation_open_CV.ipynb ##### 
- Image variable and image id reference 🐑💬 There is no need to copy the entire array because the image is a dataset of array you can use Id references, PIL images important properties, CV2 with image rotate and flip functions, CV2 crop image, CV2 draw a rectangular inside image and CV2 putText, cv2 image flip.

```
flip = {
         "ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,
         "ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,
         "ROTATE_180":cv2.ROTATE_180
       }
```

##### 2.3.2_Histogram_and_Intensity_Transformations.ipynb ##### 
- Numpy arrays management library, matplotlib plot, subplot, histogram of image and arrays, bar chart, greyscale, histogram sequence calculation from cv2.calcHist 🐑💬 Image identification by intensity distribution you need to select correct channel or ranges, image input scales from [0, 1] to [0, 255] or *[-255, 255], CV rectangular draw, image brightness and fundamental. 🐑💬 Linear modulo image for medical investigation, contrast adjustment 🧸💬 We do not compare the image's intensity amplitudes but the pattern or we can transform them into the frequency domain using a short-time furrier transform. Histogram equalization 🐑💬 It is a matrixes linear scales method. Thresholding and simple segmentation by response range of image input representing, transform image from its scales to another scales by conditions or domain variance.  

##### Alpha beta ranges array for image pixel contrast #####
🐑💬 In linear scales tangent line indicates how much of the target values different can be added or subtracted to maintain the meaning thresholds Y = mx + C.
```
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
```

##### Image binary ratios #####
🐑💬 cv2.THRESH_BINARY is the original idea to reduce the number of different inputs for the compression method and restore by intensity scales on the client.
<br>🧸💬 cv2.THRESH_TRUNC is the enchant matrixes of no-meaning pixels.</br>
<br>🐐💬 cv2.THRESH_OTSU is to solve the problem about unfairs selected representing matrix.</br>
```
cv2.THRESH_BINARY
cv2.THRESH_TRUNC
cv2.THRESH_OTSU
```

* 2.4.1_Gemetric_trasfroms_PIL.ipynb
* 2.4.2_Gemetric_trasfroms_OpenCV.ipynb
* 2.5.1_Spatial_Filtering-PIL.ipynb
* 2.5.2_Spatial_Filtering.ipynb
* 3.1_linearclassiferPytorch.ipynb
* AI Capstone Project with Deep Learning.ipynb

#### Basics Nuerons Networks methodology ####
* DL0101EN-1-1-Forward-Propgation-py-v1.0.ipynb
* DL0101EN-3-1-Regression-with-Keras-py-v1.0.ipynb
* DL0101EN-3-2-Classification-with-Keras-py-v1.0.ipynb
* DL0101EN-4-1-Convolutional-Neural-Networks-with-Keras-py-v1.0.ipynb
* DL0321EN-1-1-Loading-Data-py-v1.0.ipynb
* DL0321EN-2-1-Data-Preparation-py-v1.0.ipynb
* DL0321EN-3-1-Pretrained-Models-py-v1.0.ipynb
* DL0321EN-4-1-Comparing-Models-py-v1.0.ipynb
* DL0321EN-4-1-Comparing-Models-py-v1.ipynb

#### Multiple of adaptive neurons networks ####
* ML0120EN-1.1-Review-TensorFlow-Hello-World.ipynb
* ML0120EN-1.2-Review-LinearRegressionwithTensorFlow.ipynb
* ML0120EN-1.4-Review-LogisticRegressionwithTensorFlow.ipynb
* ML0120EN-2.2-Review-CNN-MNIST-Dataset.ipynb
* ML0120EN-3.1-Reveiw-LSTM-basics.ipynb
* ML0120EN-3.2-Review-LSTM-LanguageModelling.ipynb
* ML0120EN-4.1-Review-RBMMNIST.ipynb
* ML0120EN-Eager_Execution.ipynb

#### Applied sciences for basics neurons networks and adaptation ####
* CNN.ipynb
* Neural_Network_RELU_vs_Sigmoid.ipynb
* Simple_Neural_Network_for_XOR.ipynb
* Support_Vector_Machines_vs_Vanilla_Linear_Classifier.ipynb
* Training_a_Neural_Network_with_Momentum.ipynb
* use-objectdetection-faster-r-cnn.ipynb
* Data_Augmentation.ipynb
* Digit_Classification_with_Softmax.ipynb
* FinalProject.ipynb
* Logistic_Regression_With_Mini-Batch_Gradient_Descent.ipynb


