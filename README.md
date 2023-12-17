# Neuron-Networks-review
Review of neuron networks class material from IBM data science

## Jupiter notebook files ##
#### Images processing ####
##### 1.0_load_and_display_data.ipynb ##### 
- Initial dataset from simple images using IO, PIL, matplotlib and pandas.
##### 2.1_data_loader_PyTorch.ipynb ##### 
- Prepare the dataset by creating folders for each type of target categorized for AI-supervised learning with sample sizes and ratios.
- Supervised training, sample datasets, and labels we can do by configuration file or folder name, and the folder name is managed easily.
- 👧💬 In TensorFlow ImageDataSet Generator support configuration file and folder name, simply generate indexes of objects and labels and you can specific number of record or random its output continue until the end there is a dataset class object inheritance.

##### Custom dataset #####
👧💬 The same as in Python - TensorFlow / Tensorflow-lite, PyTorch - Tensorflow / Tensorflow-lite

```
class CustomDataset(Dataset):
    """custom dataset."""

    def __init__(self, index_filename):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...

    🧸💬 Possible working with internal function for validation or random numbers.
    def __Thee_axis_alignement__(self, dX, dY, dZ):
        ...
```

##### 2.1.2_Images_with_python_library_CV.ipynb ##### 
- Images concatenation, file string attribute manipulation, important image property size, identities, and dimension, image plotting and negative image, image label system conversion using CV2, save and read images using CV2, colour label conversion grey scales, colour scales, negative colour, RGBA, RGB, *RGBY system. Image plot and sub-images plot, indexing and image cropping, image array values copy for parallel process images system, image colour channels and CV2 image colour system conversion sample BGR_to_RGB.
- 🐑💬 In printing we need to convert RGBA to RGBY because of information in Y channel, converted back to RGBA is not the same think about Y channel is arrays of information.
- 🦭💬 Someone experiments on RGBY images the same as PDF and information beware of misused he passed but teacher pattern about the image standard is RGB or RGBA.

##### Images and Camera library using CV2 #####
```
"""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""
def update( frame ):

    ret, frame = vid.read()
    
    if ( ret ):
        
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )  
        
        o_image = tf.image.flip_up_down( frame )
        o_image = tf.keras.utils.array_to_img( o_image )
        o_image = tf.image.resize( o_image, [64, 64] )
        o_image = tf.cast( o_image, dtype=tf.int32 )
        
        image = filters( o_image )
        
        fig.axes[1].clear()
        plt.axis( 'off' )
        
        coords = find_image_countour( image )
        
        fig.axes[1].imshow( image )
        fig.axes[1].axis( 'off' )
        fig.axes[1].grid( False )
        fig.axes[1].plot( Y, X )
        fig.axes[1].fill( Y, X, "c")
        fig.axes[1].set_xlim( 0, 64 )
        fig.axes[1].set_ylim( 0, 64 )

        img_buf = io.BytesIO()
        plt.savefig( img_buf, format = "png" )
        
        image = Image.open( img_buf )
        im.set_array( image )

    return im
```
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection.jpg">
</picture>

##### 2.2.1_basic_image_manipulation_PIL.ipynb ##### 
- Introduce Python array management library Numpy, copy image array for new identity of same value for parallel images processing, object Id value and references, working with images as an array, image array properties, using Numpy array function to working with image sample flip and ImageOps from PIL, image array cropping using Numpy and Python array fundamental and value update, PIL image draw with fill colour function, * image font from PIL ImageFont, image overlay with cropped image.

##### PIL ImageOps rotation for image.transpose() function #####
🧸💬 It is Enumurator number mapping of functions and you can find some usages such as cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV that depends on the mode number they are setting.
[Jump to]([https://github.com/jkaewprateep/LittleLemonAPI](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#ops-codes))


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

##### 2.5.2_Spatial_Filtering.ipynb ##### 
- Linear Filtering, Filtering Noise, Gaussian Blur, and removal 🐑💬 This is used in printing technology. Image Sharpening, Edges, Sobel, Linear scales conversion cv2.convertScaleAbs, Sum derivative image cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) 🐑💬 Image depth enchantment method. Image median blur cv2.medianBlur and thereshold function parameter cv2.threshold. 

##### Linear Filtering #####
```
# Create a kernel which is a 6 by 6 array where each value is 1/36
kernel = np.ones((6,6))/36

# Filters the images using the kernel
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)
```

##### GaussianBlur #####
```
image_filtered = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
```

##### Sobel function #####
🐑💬 Derivative of image in X or Y direction, both X and Y derivative is complexed.
```
ddepth = cv2.CV_16S
# Applys the filter on the image in the X direction
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
```

##### Threshold Function Parameters #####
🐑💬 For fairs selected and meaning pixels.
```
# Returns ret which is the threshold used and outs which is the image
ret, outs = cv2.threshold(src = image, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
```

##### 3.1_linearclassiferPytorch.ipynb ##### 
- Pytorch dataset from image initialize object, Transform Object and Dataset Object, Convert array inputs to tensor object, Convert tensors object to Pytorch dataset object, SoftMax custom module, Optimizer, Citerian or loss value optimization functions and historical record, Data Loader and Train Model.

##### Pytorch dataset object class #####
```
class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="/resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:10000] #Change to 30000 to use the full test dataset
            self.Y=self.Y[0:10000] #Change to 30000 to use the full test dataset
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)    
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        
        image = Image.open(self.all_files[idx])
        image = image.resize((28, 28))
        y=self.Y[idx]
          
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
```

##### Convert array inputs to tensor object #####
```
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform =transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, std)])
```

##### Convert tensors object to Pytorch dataset object #####
```
dataset_train=Dataset(transform=transform,train=True)
dataset_val=Dataset(transform=transform,train=False)
```

##### SoftMax custom module #####
```
class SoftMax(nn.Module):
    
    # Constructor
    # 1. The input size should be the size_of_image
    # 2. You should record the maximum accuracy achieved on the validation data for the different epochs
    # For example if the 5 epochs the accuracy was 0.5, 0.2, 0.64,0.77, 0.66 you would select 0.77.
    
    # 🧸💬 Linear layer normally we working with same input and output the question does not tell you
    # about target size but as it is linear the activation and the output layer can caragorize objects
    # you also can apply binary catagorized or sphase catagorizes.
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        z = self.linear(x)
        return z

model = SoftMax( input_dims, output_dims )
```

##### Optimizer #####
```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
```

##### Citerian or loss value optimization functions and historical record #####
```
criterion = nn.CrossEntropyLoss()
```

##### Data Loader #####
```
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=50)
```

##### Train Model #####
```
# batch size training:5
n_epochs = 5
loss_list = []
accuracy_list = []
N_test = len(dataset_val)

def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28 * 3))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            
        correct = 0
        # perform a prediction on the validationdata  
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28 * 3))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)

print( max( accuracy_list ) )
```

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
* AI Capstone Project with Deep Learning.ipynb

## References ##
| Number of references | Target sources | Description |
|-----:|---------------|---------------|
|     1| https://en.wikipedia.org/wiki/Kernel_(image_processing) | Linear filtering image matrixes and useful method |
|     2| IBM Data Science's Jupiter NoteBook | IBM Data Science course on Coursera |

## Applications ##

#### Games simulation ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection_in_games_environment.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection_in_games_environment.jpg">
</picture>

#### Ops codes ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/error_ops_codes.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/error_ops_codes.jpg">
</picture>
