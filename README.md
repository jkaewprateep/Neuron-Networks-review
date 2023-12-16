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
- 🐑💬 cv2.THRESH_BINARY is the original idea to reduce the number of different inputs for the compression method and restore by intensity scales on the client.
- 🧸💬 cv2.THRESH_TRUNC is the enchant matrixes of no-meaning pixels.
- 🐐💬 cv2.THRESH_OTSU is to solve the problem about unfairs selected representing matrix.
```
cv2.THRESH_BINARY
cv2.THRESH_TRUNC
cv2.THRESH_OTSU
```

##### 2.4.1_Gemetric_trasfroms_PIL.ipynb ##### 
- PIL image rescales, image rotation, image as array number identification, image greyscales, singular value decomposition 🐑💬 It is the decomposing of the image into diagonal and vector images, it has the same properties as images for identity identification and process if you are using correct by its design and you can combined them after operation. This way to prevent you from having original images but you still can process them. Regeneration of different scale images for information. 🐐💬 We can apply this algorithm to multiple layers of image.

```
U, s, V = np.linalg.svd(im_gray , full_matrices=True)
```

##### 2.4.2_Gemetric_trasfroms_OpenCV.ipynb ##### 
- CV2.resize image resize using CV2, image translation, cv2.getRotationMatrix2D image rotation, image colour label transformation cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), noise image generation, singular value decomposition or linear alignment decomposition 👧💬 Select correct ratios noise supposed to be random not aligned with the image linear ratios. 

##### Image resizing interpolation method #####
- source is image input.
- dsize is the target dimension.
- fx is the scale factor of horizontal pixels.
- fy is the scale factor of vertical pixels.
- interpolation is target interpolation method.
```
cv2.resize(toy_image, None, fx=2, fy=1, interpolation = cv2.INTER_NEAREST )

cv2.INTER_AREA: Area corresponding method.
cv2.INTER_CUBIC: Matrix corresponding method.
cv2.INTER_LINEAR: Linear scale manipulation method. This is the default interpolation technique in OpenCV.
```

##### Image translation #####
- 🐑💬 Translation image by image shifted or image phase, image phase translation is required when you need to compare two graphs of their alignment.
- 🦭💬 For linear comparison we do not need to have graphs collide but similarity and significant values. 🎛🎚🌐   
```
tx = 100
ty = 0
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_image = cv2.warpAffine(image, M, (cols, rows))
```

##### 2.5.1_Spatial_Filtering-PIL.ipynb ##### 
- Linear filtering PIL ImageFilter, Gaussian blur ImageFilter.GaussianBlur, or ImageFilter.GaussianBlur(n_size), image sharpening ImageFilter.SHARPEN, Ridge or edge detection ImageFilter.EDGE_ENHANCE and ImageFilter.FIND_EDGES, and ImageFilter.MedianFilter.

##### Create custom image filters #####
🐑💬 The same as stride matrix or manipulation matrix in image convolution layers.
```
# Create a kernel which is a 5 by 5 array where each value is 1/36
kernel = np.ones((5,5))/36
# Create a ImageFilter Kernel by providing the kernel size and a flattened kernel
kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())
```

##### Image sharpen custom filters #####
Ref[0]: https://en.wikipedia.org/wiki/Kernel_(image_processing)

##### Common Kernel for image sharpening, ImageFilter.SHARPEN #####
```
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
kernel = ImageFilter.Kernel((3,3), kernel.flatten())
```

##### Ridge or edge detection, ImageFilter.EDGE_ENHANCE and ImageFilter.FIND_EDGES #####
```
kernel = np.array([[ 0,-1, 0], 
                   [-1, 4,-1],
                   [ 0,-1, 0]])
kernel = ImageFilter.Kernel((3,3), kernel.flatten())

or

kernel = np.array([[-1,-1,-1], 
                   [-1, 8,-1],
                   [-1,-1,-1]])
kernel = ImageFilter.Kernel((3,3), kernel.flatten())
```

🐑💬 Another way is to use median filters for edge selection.
```
ImageFilter.MedianFilter
```

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


