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
[Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Imges-centre-detection)
- 👧💬 🎈 DataLoader is custom data presenter and you can write custom function inside as in the picture.
[Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#DataLoader)

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


##### 2.2.1_basic_image_manipulation_PIL.ipynb ##### 
- Introduce Python array management library Numpy, copy image array for new identity of same value for parallel images processing, object Id value and references, working with images as an array, image array properties, using Numpy array function to working with image sample flip and ImageOps from PIL, image array cropping using Numpy and Python array fundamental and value update, PIL image draw with fill colour function, * image font from PIL ImageFont, image overlay with cropped image.

##### PIL ImageOps rotation for image.transpose() function #####
🧸💬 It is Enumurator number mapping of functions and you can find some usages such as cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV that depends on the mode number they are setting.
[Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#ops-codes)

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
- Numpy arrays management library, matplotlib plot, subplot, histogram of image and arrays, bar chart, greyscale, histogram sequence calculation from cv2.calcHist.
- 🐑💬 Image identification by intensity distribution you need to select correct channel or ranges, image input scales from [0, 1] to [0, 255] or *[-255, 255], CV rectangular draw, image brightness and fundamental.
- 🐐💬 Another tip for image identification is image scales, linear scales, or images with channel concatenation.
- 🐑💬 Linear modulo image for medical investigation, contrast adjustment.
- 🧸💬 We do not compare the image's intensity amplitudes but the pattern or we can transform them into the frequency domain using a short-time furrier transform. Histogram equalization.
- 🐑💬 It is a matrixes linear scales method. Thresholding and simple segmentation by response range of image input representing, transform image from its scales to another scales by conditions or domain variance.
- 🦭💬 I revealed to you some of the mathematics discovered images in the computer we are using most of them are arrays of linear scales and when differentiated the linear variables value that is not changing over time or domain will be removed resulting in something moving on the screen. We can have edge detection but saved of the calculation process we consider moving objects as our eyes and senses.

##### Alpha beta ranges array for image pixel contrast #####
🐑💬 In linear scales tangent line indicates how much of the target values different can be added or subtracted to maintain the meaning thresholds Y = mx + C.
```
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
```

##### Image binary ratios #####
- 🐑💬 cv2.THRESH_BINARY is the original idea to reduce the number of different inputs for the compression method and restore by intensity scales on the client.
- 🧸💬 cv2.THRESH_TRUNC is the enchant matrixes of no-meaning pixels.
- 🐐💬 cv2.THRESH_OTSU is to solve the problem about unfairs selected representing matrix.
- 👧💬 🎈 cv2.THRESH_BINARY is a direct conversion when presenting pixels matrix in specific ranges of ratios present in one value and reverse as a binary image.
- 👧💬 🎈 cv2.THRESH_TRUNC find a presenting value of pixels matrix with neighbors and re-scales.
- 👧💬 🎈 cv2.THRESH_OTSU combined both methods to solve the unfairs selection presenter matrix.  
```
cv2.THRESH_BINARY
cv2.THRESH_TRUNC
cv2.THRESH_OTSU
```

##### 2.4.1_Gemetric_trasfroms_PIL.ipynb ##### 
- PIL image rescales image rotation, image as array number identification, image greyscales, singular value decomposition.
- 🐑💬 It is the decomposing of the image into diagonal and vector images, it has the same properties as images for identity identification and process if you are using the correct design and you can combine them after operation. This way to prevent you from having original images but you still can process them. Regeneration of different scale images for information.
- 🐐💬 We can apply this algorithm to multiple layers of image.
- 🦭💬 The equation is to reduce the form of the matrix using similarity using the Eigant value and Eigant vector, decomposed of images arrays when preserved of their property.
```
U, s, V = np.linalg.svd(im_gray , full_matrices=True)
```

##### 2.4.2_Gemetric_trasfroms_OpenCV.ipynb ##### 
- CV2.resize image resize using CV2, image translation, cv2.getRotationMatrix2D image rotation, image colour label transformation cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), noise image generation, singular value decomposition or linear alignment decomposition.
- 👧💬 Select correct ratios noise is supposed to be random and not aligned with the image linear ratios.
- 🦭💬 For fast computation Eigant vector is a vector representing of image and the Eigant value is similarity, The Eigant value can reduce form and can perform calculations as the original same as image fast track change.

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
- Linear Filtering, Filtering Noise, Gaussian Blur, and removal.
- 🐑💬 This is used in printing technology. Image Sharpening, Edges, Sobel, Linear scales conversion cv2.convertScaleAbs, Sum derivative image cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
- 🐑💬 Image depth enchantment method. Image median blur cv2.medianBlur and thereshold function parameter cv2.threshold. 

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
🐨🎁🎵🎶 Sample of Tensorflow and Pytorch DataSet.
[Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Tensorflow-and-Pytorch-Custom-DataSet)

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

#### Neurons Networks recaps ####
- 🐑💬 A single-layer perception is simply a response as reflection, observing a metal plate that shrinks or expands when heats distribution but not a knee knot hammer.
- 👧💬 🎈 Radius Basics is a single complex network that seems like redundancy work such as electrical switches or water tap control.
- 🧸💬 Multi-layer perception and recurrent networks are famous networks because many experiments published and they can be explained by non-complex features. There are many types and various of the number of layers and nodes when recurrent networks have multiple logical gates to select.
- 🐑💬 This is a known type of network that is AND OR networks logic or XOR logic we are using in electrical instruments such as current observing or pass-through logic for input from digits panels.
- 🧸💬 Multi-perception networks can simulate logical gates but they require training or similar networks for the same function this is one weakness they are improved by LSTM recurrent networks that can learn with logical gate pre-build.
- 🐐💬 Hopped fields or stars networks had working nodes as ring networks and in one of the node failures they copied the same function for the most appropriate selected number of nodes target.
- 🦭💬 Bolzedmann machines, select one side of the network it can calculate of multiple works in the same time. The equivalent of microscopics create work output on another side.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xenonstack-neural-network-architecture-3-1-1.webp">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xenonstack-neural-network-architecture-3-1-1.webp">
</picture>
<br>Ref[2]: https://www.xenonstack.com/blog/artificial-neural-network-applications</br>
</p>

#### Linear behavior of neurons networks node ####
🐑💬 Nueron networks are supposed to be complex and vary by multiple variables but the calculation will be too difficult and we cannot prove of its results behavior. For larger networks, we simulate each node's present value of node bias and weight Y = mx + C. This method allowed us to calculate the entropy network value at one edge of the network, from action to network until reaching one edge of the network presents a label or target function as a result. You can initial value of the weight and bias or use pre-trained values that are built with the commercial logic board and electronics path. Arduino and many logic boards make training easier and they build in pre-trained networks if you select the correct path and method.  
```
import numpy as np # import Numpy library to generate 

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
```
🐑💬 Linear behavior of neuron network layer. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Weight-response-calculation)
```
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
z_11 = round( z_11, 3 )

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
```
🐑💬 Activation function, linear exponential is fast growth with the value change from this discovery make it to be true, some discovery channel.
```
a_11 = 1.0 / (1.0 + np.exp(-z_11))
```
🐑💬 The same as training history, loss value and pre-defined value in historical small_network['layer_1']['node_1']['weights'] is sum layer weight and small_network['layer_1']['node_1']['bias'] is sum layer bias. The compute_weighted_sum function is a summary of the network's bias and network weight function, the actual training function should sum of new approaches with current by drawing from the estimation function or loss estimation value for the gradient decent method.
[Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Cylindical-Problem)
```
### type your answer here
small_network = initialize_network(5, 3, [3, 2, 3], 1)

node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))
```
🧸💬 The node activation function is to create contrast differentiation of input values with specific conditions or mathematical methods. The fastest way is to use an absolute function that is because it is a linear function, the sample of an absolute function is an alphabet ruler, and single-layer networks have something direct to answer such as a push button.
```
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

### type your answer here
node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))
```
🦭💬 Create simple networks from custom DenseLayer in Tensorflow, identifying nodes and initial values.
```
### create a network

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.initializer = tf.keras.initializers.Identity()
        self.kernel = self.add_weight(shape=[int(input_shape[-1]),
        self.num_outputs],
        initializer = self.initializer,
        trainable=True)
        self.kernel = tf.cast( self.kernel, dtype=tf.float32 )
        self.weight = self.add_weight(shape=[int(input_shape[-1]),
        self.num_outputs],
        initializer = tf.zeros_initializer(),
        trainable=True)

    def call(self, inputs):
        result = tf.matmul( tf.cast( inputs, dtype=tf.float32 ), self.kernel )
        result = tf.math.add( result, self.weight )
        result = tf.cast( result, dtype=tf.int32 )

        return result

word_char_length = 12
layer = MyDenseLayer(word_char_length)
input_value = tf.constant(
[[ 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], shape=(12, 12), dtype=tf.float32)

result = layer( input_value )

print( result )
```
🦭💬 Create new networks from custom DenseLayer in Tensorflow, identifying nodes and initial values.
```
### create another network

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
        shape=[int(input_shape[-1]),
        self.num_outputs])

    def call(self, inputs):
        result = tf.matmul(inputs, self.kernel)		# , shape=(10, 10), dtype=float32)
        return result

start = 3
limit = 33
delta = 3
sample = tf.range(start, limit, delta)
sample = tf.cast( sample, dtype=tf.float32 )
sample = tf.reshape( sample, shape=( 10, 1 ) )
layer = MyDenseLayer(10)
result = layer(sample)
print( result )

### Thank you for laboratory ###
```

* DL0101EN-3-1-Regression-with-Keras-py-v1.0.ipynb
- 🦭💬 Regression problem is a solution that can find answers by substitution method within a finite set of input and output. 

#### Regression problem ####
🐐💬 We call them dependent variables and co-variants, possibly finding the relationship between input to output within specific scopes of interest. ( 🐑💬 finite because none finite we cannot find values )   

🐐💬 Using Panda library to download s CSV file into a data frame.
```
import pandas as pd

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
```
🐐💬 Using Keras to create sequential model networks with simple Dense layers, and initial of its value.
```
import keras
from keras.models import Sequential
from keras.layers import Dense

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```
🐐💬 Training model with normalized value and target strength, the input value is high variances or a small number of samples we can use normalized value to help with learning time of the networks. By expectation high variances should provide the best results because it is different but requires complex relationship too.   
```
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
```

* DL0101EN-3-2-Classification-with-Keras-py-v1.0.ipynb
- 🦭💬 In a classification problem we categorize the target by output nodes value mapping to label, logits shape, or label dimension and value. We can determine of target label by cross-entropy function sample SoftMax, CategoricalCrossentropy, BinaryCrossentropy, CosineSimilarity, etc.   

### Load standard dataset ###
🦭💬 The MNIST handwriting database tfds.image_classification.MNIST is a standard database that can be used to evaluate the performance of the networks.
Ref[4]: https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification/mnist.py
```
# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### Evaluation of the model ###
🦭💬 Evaluation of the model to examine network prediction scores by input and target output results in accuracy scores and error. High accuracy scores do not mean being the best networks but high accuracy with steady variances or sometimes they are testing the networks on unseen datasets with the same variances and provide the same or similar results in accuracy scores. The Evaluation method is faster and does not require multiple times of execution because it selects appropriate ranks for the testing dataset. 
```
# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
```

#### Load pre-trained model ####
🦭💬 From saved or a pre-defined network shape you can load target weights or weights and a configuration file into target sequential. 
```
from keras.models import load_model

pretrained_model = load_model('classification_model.h5')
```

* DL0101EN-4-1-Convolutional-Neural-Networks-with-Keras-py-v1.0.ipynb
- 👧💬 🎈 Convolution layer is matrix manipulation from target pixels and its neighbor resulting in the representation of sample matrix and it is a powerful method in computer visions and instruments. Imagine when you shifted or calculated numbers into a target single value at a time and target matrix area, they also have comparison and interaction in the same way as they are convoluted numbers with matrix transform direction into target images and restoration. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Convolution-layer)

#### Sample of creating a sequential model with convolution layer ####
👧💬 🎈 Convolution layer usage is the same as Dense and layers, you can create a custom Convolution layer from tf.keras.layers.Conv2D class. What do the custom convolution layers can do? They can perform sinusoid functions or concatenated them for the best response target weight matrix. * Multiple residual networks can perform with custom layer as recurrent networks do but are defined and present for understanding.   
```
def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model
```

#### Sample of multiple convolution layers ####
👧💬 🎈 Convolution layers can be pre-defined and respond to target input with values matrixes that are stored in the class object, multiple layers of convolution response in different input significant properties such as shape and colours ( input values ) by multiple-times running with properly matched target object shape and colours remain or significant. Larger sizes of matrix respond to the target size of an object and also the direction of scanning and stored matrix values response to target numbers in ranges because it is matrix multiplication see edge detection for more information. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#common-kernel-for-image-sharpening-imagefiltersharpen)
```
def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model
```

* DL0321EN-1-1-Loading-Data-py-v1.0.ipynb
- 🧸💬 For a simple method creating arrays of data and its label them to perform a calculation with a target function in a sequential model is possible by using Numpy library, Python library, Panda dataframe, index files, and database. You can refer to the target dataset, target file location, versions, functions and target data sources allowed to transfer of the dataset object with the same method to perform the same output. Same as some applications if we only need a strict policy to fully works on the target application but do not allowed to copy most of IT admin using security methods but file size and working policy are also applicable do not too look down this step. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Input-identity-identification-method)    

#### Create arrays input and label from images target ####
```
%%time

import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image

## 🧸💬 The os scandir is monitoring of new objects activity
## That is a good ideas when there is less than hundred of files
## you can perform file create or modify events but you do not
## need to registered that much of objects.
negative_files = os.listdir('./Negative')
negative_files

negative_images = []
for file_name in negative_files:
    if os.path.isfile(file_name):
        image_name = str(file_name).split("'")[1]
        image_data = plt.imread('./Negative/{}'.format(image_name))
        negative_images.append(image_data)
    
negative_images = np.array(negative_images)
```

* DL0321EN-2-1-Data-Preparation-py-v1.0.ipynb
- 🐐💬 Data Image Generator can perform image data manipulation and extra with the custom function they allowed tf.keras.preprocessing.image.ImageDataGenerator#preprocessing_function. In this method we can perform negative images, hue images with target functions, or differentiation of its input images by the pre-processing method. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#image-pre-process-and-compressed-commands)

#### Create ImageDataGenerator class ####
🐐💬 The instance of ImageGenerator class is more than indexes, it creates rewindable indexes with tensor shape, we can work with tensors shape as a simple tensors shape but they remain the property of item identification. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#multi-process-and-tf-agents)  
```
# instantiate your image data generator
data_generator = ImageDataGenerator()

image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
```

#### Samples from created ImageDataGenerator ####
```
## You can use this cell to type your code to answer the above question
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )

first_batch_images = image_generator.next()[0] # first batch
second_batch_images = image_generator.next()[0] # second batch
third_batch_images = image_generator.next()[0] # second batch
forth_batch_images = image_generator.next()[0] # second batch
fifth_batch_images = image_generator.next()[0] # second batch

# 🧸💬 Image calsses from image data generator.
print( image_generator.class_indices )
print( image_generator.num_classes )
print( image_generator.classes[0:24]  )

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        # image_data = third_batch_images[ind].astype(np.uint8) 
        image_data = fifth_batch_images[ind]
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Third Batch of Concrete Images' + " " + str(  "") ) 
plt.show()
```

#### Results ####
🐐💬 The results explain about the number of image input ( manipulate value we found a bug in previous MNIST versions ), class of the target we can identify by arrays, name of folder or indexes files and function and number of class the ImageGenerator contain, and sample of first 24 indexes. 🐑💬 This is for training purposes working actual we need to use distributed data use the shuffles. 
```
Found 40000 images belonging to 2 classes.
{'Negative': 0, 'Positive': 1}
2
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```
 
* DL0321EN-3-1-Pretrained-Models-py-v1.0.ipynb
- 🐑💬 Pre-trained model is also the training model with a test dataset prepared to test with the actual dataset we need to work with the sequential model and work inputs prepared for the variances. Pre-trained sequential models are available and commonly used with no extra costs because professionals in this field build for us.

#### Create a sequential model from pre-trained model ####
🐑💬 Do not forget to add Sequential and Dense layers when output prints the same as class debugging inheritance and target output usage.
```
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

model = Sequential()
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
```
#### Training sequential model ####
🐑💬 It is important to compile before training to initial value back to their settings, sometimes steps per epoch can change for overlap results in the same set as circular with small ratios for better results.
```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
```

#### Save the result from work ####
```
## 🧸💬 You can safe model check points, model with parameters and all working solution
## Please see the reading manual select for correct purpose. Reshaers need to use logs 
## for backward result identification if not joined the labs.

model.save('classifier_resnet_model.h5')
```

* DL0321EN-4-1-Comparing-Models-py-v1.0.ipynb
* DL0321EN-4-1-Comparing-Models-py-v1.ipynb
- 🦭💬 Create a sequential model from a pre-trained model, one method to provide them the data sources without being exposed or trapped by network package identification is by the dataset but some bad man try to use it as a hacker step. Inside the dataset can contain multiple functions to manipulate and sort data when they are stored as sequences of binary numbers or encrypted messages. 🦭💬 We noticed with someone running the dataset or later running. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#data-conditions)

#### Create a sequential model from pre-trained model ####
```
model = Sequential()
model.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.save('classifier_resnet_model.h5')
```

#### Model Summary ####
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 512)               14714688  
                                                                 
 dense (Dense)               (None, 2)                 1026      
                                                                 
=================================================================
Total params: 14,715,714
Trainable params: 14,715,714
Non-trainable params: 0
_________________________________________________________________
```

#### Model Evaluation ####
```
scores = model.evaluate(validation_generator)
```

#### Results ####
```
[0.7199065089225769, 0.49994736909866333]
```

#### Model Prediction ####
```
predict = model.predict_generator(validation_generator, steps = 100)
```

#### Results ####
```
[[0.38565585 0.6143441 ]]
```

#### Multiple of adaptive neurons networks ####
* ML0120EN-1.1-Review-TensorFlow-Hello-World.ipynb
- 🧸💬 Fundamental methods for TensorFlow, create tf.tensors from tf.constant, multiple-dimensions arrays operations, and tf.variable. There are steps of calculation and types of variables, output from multiplication process or process no reverse form can be stored in tf.contant and tf.variablle when the calculation step is in tf.variable form. [Jump to](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#tf-variables-in-gradients-optimization)

#### TF.constant example ####
```
a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)
```

#### TF.Variable example ####
```
v = tf.Variable(0)

@tf.function
def increment_by_one(v):
        v = tf.add(v,1)
        return v
```

* ML0120EN-1.2-Review-LinearRegressionwithTensorFlow.ipynb
- 🐨🎁🎵🎶 Optimization problem critical method linear regression, optimization values from the pre-defined relationship and significant. It does not directly find the relationship but linear regression can do in small variances and using it this way is not quite correct and uses some resources. Forward-backward substitution method, feedback propagation, variance distribution, and there are many learning methods research discovered and found on public resources. Some simple methods of learning in the classroom that can find solutions for many problem and tasks within in scope of the problem set is the substitution method and feedback propagation.

#### Function and TF-Variables ####
🐨🎁🎵🎶 By substitution of a or b value each time of incremental or decrease to match the target aligns, the substitution technique can find the value numbers in a pre-defined relationship and they are using tf.Variable because some processes are not the final no reverse substitution. [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#power-series)   
```
a = tf.Variable(20.0)
b = tf.Variable(30.2)

def h(x):
   y = a*x + b
   return y
```

#### Gradient Tape - regression problem solution finding #### 
🐨🎁🎵🎶 Gradient Tape optimization is used in some tasks that require accumulated value evaluation or assignment feedback value from summing that is a faster calculation when working with large datasets and in larger platform distributed they are using this method for remote running execution because of accuracy work same as precision determine. 
[Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/tree/main?tab=readme-ov-file#tf-variables-in-gradients-optimization), Remote execution [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#remote-execution-sample)
```
learning_rate = 0.01
train_data = []
loss_values =[]
a_values = []
b_values = []
# steps of looping through all your data to update the parameters
training_epochs = 200

# train model
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_predicted = h(train_x)
        loss_value = loss_object(train_y,y_predicted)
        loss_values.append(loss_value)

        # get gradients
        gradients = tape.gradient(loss_value, [b,a])
        
        # compute and adjust weights
        a_values.append(a.numpy())
        b_values.append(b.numpy())
        b.assign_sub(gradients[0]*learning_rate)
        a.assign_sub(gradients[1]*learning_rate)
        if epoch % 5 == 0:
            train_data.append([a.numpy(), b.numpy()])
```

* ML0120EN-1.4-Review-LogisticRegressionwithTensorFlow.ipynb
- 🦭💬 Difference between linear regression and logistic regression is that the target output is continuous, two it is logistics it does not require a label and three logistics can be adaptive to the target platform easily. See the sample of logistics output or logits shape and see from example is easy to understand than reading word because of once I read it for the first time I have the same question, how it not always require label but they can communications [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/tree/main?tab=readme-ov-file#image-pre-process-and-compressed-commands) 

#### Logistics regression ####
🦭💬 There is no logits label or logits value different from the label than the regression label solution.
```
# Three-component breakdown of the Logistic Regression equation.
# Note that these feed into each other.
def logistic_regression(x):
    apply_weights_OP = tf.matmul(x, weights, name="apply_weights")
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
    return activation_OP
```

* ML0120EN-2.2-Review-CNN-MNIST-Dataset.ipynb
- 👧💬 🎈 Deep learning is an artificial network ability to learn and adaptative to conditions and rules that are more complex than single or few layers. There is something called the degree of transformation because it is difficult to define deep-learning networks when the number of layers does not specify the function costs. The differentiate and transform function identified value and identity pattern and similarity equation can define how much of two or more equation similarity sample Eigant value and Eigant vector for similarity comparision.    

#### Forward function ####
👧💬 🎈 Fundamental forward function represents linear function behavior, small value changes indicate the summing add of the value until the multiply variable has the same indicates will move to multiply. In some algorithms, they make it fast by starting by multiplying value and then increasing the summing value when indicated when using the loss value estimation function because it can be summing for smaller value decimals and faster reach the nearest point. ( 💃( 👩‍🏫 )💬 Sample momentum loss value estimation function ) 🐑💬 In programming we see W and b are the same type of tf.variable and they are updated by each iteration of the training process assigners are from the loss value estimation function and the optimizer function is recorder. [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#customer-loss-value-estimation-function)
```
def forward(x):
    return tf.matmul(x,W) + b
```

#### SoftMax function ####
👧💬 🎈 By summary of all input as an array with the same size to 1.0 for easy to identify the most indicates from overall. There are more entropy functions but softmax is famous because it can combine with other functions to identify significance and sometimes requires a function to identify identical from different input from the same logits shape such as output to binary or compressed command for communication or error correction function. [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#image-pre-process-and-compressed-commands) 🐑💬 Thinking about networks circuits balance or impedance circuits implement into the logic simulation.
```
vector = [10, 0.2, 8]
softmax = tf.nn.softmax(vector)
```

#### Cost function ####
👧💬 🎈 Cost function or criterion when considering composite of gradient value optimization, or loss value estimation function. It can be any function working with a finite set of data and select for suitable job see the target loss value estimation function sample is Y-new - Y [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#customer-loss-value-estimation-function)
```
def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))
# addition of 1e-10 to prevent errors in zero calculations

# current loss function for unoptimized model
cross_entropy(y_train, model(x_train)).numpy()
```

#### Activation functions ####
👧💬 🎈 Activation function is a designer by ratios or conditions to target output of each node in each layer, it can be linear, sinusoidal, conditions, or any function. An example of a miss use of the activation function is some missing value or too small or too large of numbers that can create -nan or 0 value of target optimize in the network running process.
```
# ReLU activation function
def h_conv1(x): return(tf.nn.relu(convolve1(x)))

# SoftMax activation function
def y_CNN(x): return tf.nn.softmax(fc(x))
```

#### Network layers ####
👧💬 🎈 Fundamental of fully connected node and possible to create multiple operations node or internal function node. In the sequential model, you do not need to create internal function layers but you can use sinusoid to connect layer output to the next layer input in scope by specific target layer. To specific target layer is possible do it by Tensorflow, Pytorch, and TF-learn.
See the example of internal functions [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#internal-function)
```
# Fully connected layer
def fcl(x): return tf.matmul(layer2_matrix(x), W_fc1) + b_fc1

# convolution layer
def convolve1(x):
    return(
        tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# convolution layer
def convolve2(x): 
    return( 
    tf.nn.conv2d(conv1(x), W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
```

#### Interpolate function from convolution2D function ####
🦭💬 Some materials are mistyped by accident just add the convolution2D with target output logits to create a correct function with function result expecting.
```
# ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
# 🧸💬 That is because the convolve1 is only covolution layer 
# It requried target shape when you not reply the error message but 
# expectation input outcomes.
# The errors message is happen at time of the event and it required 
# intention to forward.
ActivatedUnits = convolve2(np.asarray(sampleimage).reshape((28,28,1,1)))
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
```

* ML0120EN-3.1-Reveiw-LSTM-basics.ipynb
- 🐨🎁🎵🎶 Re-current artificial networks work the same as computation units that compute the same data output from the previous process and continue until now ```current process * process of t - 1```. The re-current artificial networks can be significant data of the current process and will be used in the next iteration as named re-current artificial networks. During the process performed by internal logical gates and selected algorithms the re-current artificial networks select to forget ( repeating data ), memorize ( significant data ), and operate previous data with current data. By multiple or more iteration processes the re-current artificial networks produced data from significant processes. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#basics-lstm-layer-in-object-in-image-detection )

#### Explaining LSTM behavior in GAME AI object screen detection ####
🐨🎁🎵🎶 IN a plain paper something moving creates a difference from stability, matrixes on the left handside are a simulation of number 2 moving from left to right, and on the right are the results. Where is number 2 located on a plain text { 1, 2, 3, 4, 5, 6, 7, 8, 9 } and how stable does it turn { { 0 }, { 0, 0 }, { 0, 0, 0} } continuing to mean same speed velocity because of no change in distance than previous significant will be forgotten. ( 👧💬 🎈 How slow speed catch up high speed, and how to know it process? )       
```
[ 1 1 1 1 1 1 1 1 1 1 ] ==> [ 0 0 0 0 0 0 0 0 0 0 ]
[ 1 2 1 1 1 1 1 1 1 1 ] ==> [ 0 1 1 1 1 0 0 1 0 ]
[ 1 1 2 1 1 1 1 1 1 1 ] ==> [ 0 0 0 2 2 0 0 2 2 ]
[ 1 1 1 2 1 1 1 1 1 1 ] ==> [ 0 0 3 3 3 0 3 0 0 ]
[ 1 1 1 1 2 1 1 1 1 1 ] ==> [ 0 0 4 0 4 0 4 4 0 ]
[ 1 1 1 1 1 2 1 1 1 1 ] ==> [ 0 0 0 5 5 0 0 5 5 ]
[ 1 1 1 1 1 1 2 1 1 1 ] ==> [ 6 0 0 6 6 6 0 6 0 ]
[ 1 1 1 1 1 1 1 2 1 1 ] ==> [ 0 7 0 0 0 7 7 7 0 ]
[ 1 1 1 1 1 1 1 1 2 1 ] ==> [ 0 8 0 0 0 8 0 0 8 ]
[ 1 1 1 1 1 1 1 1 1 2 ] ==> [ 9 9 0 0 0 0 0 9 9 ]
```

#### LSTM input-output ####
🐨🎁🎵🎶 Because the LSTM network layer is a re-current network process, there is an input to provide the output, final_memory_state, and final_carry_state. You can select to work with input or input with carry state because the LSTM layer is defined as a class object with running. The output from LSTM layer can be the final result or the next layer input when carrying flags is an internal state of the LSTM layer and you can transfer carrying flags for the learning method between LSTM layers same as the multiple-LSTM layers process. Sometimes final_memory_state with final_carry_state or flags are determined of the process running and the significance of the process owner when there are multiple LSTM layers from multiple nodes. Yesterday someone talked about governance artificial networks these are some requirements when the fully connected layer has segmentation and LSTM can have carrying flags. 
```
output, final_memory_state, final_carry_state = lstm(inputs)
```

#### STACKED LSTM ####
🐨🎁🎵🎶 There are multiple LSTM layers in the sequential model but stacked LSTM is something different, it is built into the same unit and works with internal variable transfer within the new re-current network and works as the same recurrent network layer. This technique works with module networks because they are patterns as block codes different from residual networks to perform the same tasks again with the same network because it is a recurrent network. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#input-identity-identification-method )
```
cell1, cell2 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)
cells.append(cell2)

stacked_lstm =  tf.keras.layers.StackedRNNCells(cells)
lstm_layer= tf.keras.layers.RNN(stacked_lstm ,return_sequences=True, return_state=True)
```

* ML0120EN-3.2-Review-LSTM-LanguageModelling.ipynb
- 🦭💬 Application in words embedding and sequential inputs, in LSTM networks application the attention networks determine of the next phase of word presenting as attention scores by overall experience, language models, patterns, or configuration. [Jump To](https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#sentence-words-embedding---attention-networks) There is a method to create vector input from words in a sentence called word2vec [Jump To]( https://www.tensorflow.org/text/tutorials/word2vec ). The next process from vector input is create a mix of possibility and learning patterns by attention networks starting as My name is ____ . There are a few examples of attention words in the dataset are { Dekdee, Ploy, Ji, Noon, Praw, ... } when the iteration process with the attention network provides different values for each item.  🧸💬⁉️ The highest score from attention networks is selected which is _____.

#### LSTM networks training ####
🦭💬 By the result of the word embedding layer creation of input for the LSTM network training process, 
```
# Reads the data and separates it into training data, validation data and testing data
raw_data = ptb_raw_data(data_dir)
train_data, valid_data, test_data, vocab, word_to_id = raw_data

# word2vec has a built-in function in Tensorflow as in the Jump To link.

# Define the Gradient variables for the learning process
# Create a variable for the learning rate
lr = tf.Variable(0.0, trainable=False)
optimizer = tf.keras.optimizers.SGD(lr=lr, clipnorm=max_grad_norm)

# By regression learning method create a tape object for record of the result from the learning process.
with tf.GradientTape() as tape:
    # Forward pass.
    output_words_prob = model(_input_data)
    # Loss value for this batch.
    loss  = crossentropy(_targets, output_words_prob)
    cost = tf.reduce_sum(loss,axis=0) / batch_size

# Evaluation of the learning process same as training with optimizer and loss value estimation function. 
# Get gradients of loss with the trainable variables.
grad_t_list = tape.gradient(cost, tvars)

# Scopes by minimize and maximize values
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)

# Training process.
# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))
```

#### Create a custom LSTM model as class object ####
🧸💬 You can create a custom sequential model from the object class or tf.keras.Model class [Jump To]( https://www.tensorflow.org/api_docs/python/tf/keras/Model ) This is super easy you can remove the comment and use inherit class pattern. You can find example of time distribution model [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#input-identity-identification-method ) as many of people try to learn TensorFlow asking how to make Integrate number or differentiate sequence numbers. His past contributed [Jump To]( https://stackoverflow.com/users/7848579/jirayu-kaewprateep )
```
class PTBModel(object):

    def __init__(self):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size_l1 = hidden_size_l1
        self.hidden_size_l2 = hidden_size_l2
        self.vocab_size = vocab_size
        self.embeding_vector_size = embeding_vector_size
        # Create a variable for the learning rate
        self._lr = 1.0
        
        ###############################################################################
        # Initializing the model using keras Sequential API  #
        ###############################################################################
        self._model = tf.keras.models.Sequential()
        
        ####################################################################
        # Creating the word embeddings layer and adding it to the sequence #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            self._embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embeding_vector_size,batch_input_shape=(self.batch_size, self.num_steps),trainable=True,name="embedding_vocab")  #[10000x200]
            self._model.add(self._embedding_layer)

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM Cells. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument  of LSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A). 
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell_l1 = tf.keras.layers.LSTMCell(hidden_size_l1)
        lstm_cell_l2 = tf.keras.layers.LSTMCell(hidden_size_l2)
        
        # By taking in the LSTM cells as parameters, the StackedRNNCells function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of stacked simple cells.
        stacked_lstm = tf.keras.layers.StackedRNNCells([lstm_cell_l1, lstm_cell_l2])

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.  
        # In step 2,  second word of each of the b sentences is input in parallel. 
        # The parallelism is only for efficiency.  
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. 
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel. 

        ########################################################################################################
        # Instantiating our RNN model and setting stateful to True to feed forward the state to the next layer #
        ########################################################################################################
        
        self._RNNlayer  =  tf.keras.layers.RNN(stacked_lstm,[batch_size, num_steps],return_state=False,stateful=True,trainable=True)
        
        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = tf.Variable(tf.zeros([batch_size,embeding_vector_size]),trainable=False)
        self._RNNlayer.inital_state = self._initial_state
    
        ############################################
        # Adding RNN layer to keras sequential API #
        ############################################        
        self._model.add(self._RNNlayer)
        #self._model.add(tf.keras.layers.LSTM(hidden_size_l1,return_sequences=True,stateful=True))
        #self._model.add(tf.keras.layers.LSTM(hidden_size_l2,return_sequences=True))

        ####################################################################################################
        # Instantiating a Dense layer that connects the output to the vocab_size  and adding layer to model#
        ####################################################################################################
        self._dense = tf.keras.layers.Dense(self.vocab_size)
        self._model.add(self._dense)

        ####################################################################################################
        # Adding softmax activation layer and deriving probability to each class and adding layer to model #
        ####################################################################################################
        self._activation = tf.keras.layers.Activation('softmax')
        self._model.add(self._activation)

        ##########################################################
        # Instantiating the stochastic gradient decent optimizer #
        ########################################################## 
        self._optimizer = tf.keras.optimizers.SGD(lr=self._lr, clipnorm=max_grad_norm)

        ##############################################################################
        # Compiling and summarizing the model stacked using the keras sequential API #
        ##############################################################################
        self._model.compile(loss=self.crossentropy, optimizer=self._optimizer)
        self._model.summary()

    def crossentropy(self,y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    def train_batch(self,_input_data,_targets):
        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = self._model.trainable_variables
        # Define the gradient clipping threshold
        with tf.GradientTape() as tape:
            # Forward pass.
            output_words_prob = self._model(_input_data)
            # Loss value for this batch.
            loss  = self.crossentropy(_targets, output_words_prob)
            # average across batch and reduce sum
            cost = tf.reduce_sum(loss/ self.batch_size)
        # Get gradients of loss wrt the trainable variables.
        grad_t_list = tape.gradient(cost, tvars)
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
        # Create the training TensorFlow Operation through our optimizer
        train_op = self._optimizer.apply_gradients(zip(grads, tvars))
        return cost
        
    def test_batch(self,_input_data,_targets):
        #################################################
        # Creating the Testing Operation for our Model #
        #################################################
        output_words_prob = self._model(_input_data)
        loss  = self.crossentropy(_targets, output_words_prob)
        # average across batch and reduce sum
        cost = tf.reduce_sum(loss/ self.batch_size)

        return cost
    
    # 🧸💬 you can use @ to anything that is your own defined
    # some property match tell program to monitor registered 
    # the same as telling method has return typed or specific
    # inputs or working with target functions.
    @classmethod
    def instance(cls) : 
        return PTBModel()
```

#### Sequential model training ####
🧸💬 Iteration running process improvement of the custom model we create and evaluate from result prediction and compare. From evaluation, we can define of next approach improvement more than accuracy and > 0.2 of loss estimation value but process of solution we are looking for the answer. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#words-or-sequence-confidence-scores )
```
# Instantiates the PTBModel class
m=PTBModel.instance()   
K = tf.keras.backend 
for i in range(max_epoch):
    # Define the decay for this epoch
    lr_decay = decay ** max(i - max_epoch_decay_lr, 0.0)
    dcr = learning_rate * lr_decay
    m._lr = dcr
    K.set_value(m._model.optimizer.learning_rate,m._lr)
    print("Epoch %d : Learning rate: %.3f" % (i + 1, m._model.optimizer.learning_rate))
    # Run the loop for this epoch in the training mode
    train_perplexity = run_one_epoch(m, train_data,is_training=True,verbose=True)
    print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
    # Run the loop for this epoch in the validation mode
    valid_perplexity = run_one_epoch(m, valid_data,is_training=False,verbose=False)
    print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
# Run the loop in the testing mode to see how effective was our training
test_perplexity = run_one_epoch(m, test_data,is_training=False,verbose=False)
print("Test Perplexity: %.3f" % test_perplexity)
```

* ML0120EN-4.1-Review-RBMMNIST.ipynb
- 🐑💬 Restricted BOLTZMANN machine when the significant values update evaluation with Eigaint significant values method, before using LSTM we can determine the value to forgetting by the significant values calculation from Eigant significant values. Restricted BOLTZMANN can perform work by SOFTMAX layer output when significant value updates because there are very small updates on each iteration but learning from significant values. Restricted BOLTZMANN and collaborative filtering [Jump To]( https://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf )

#### Contrastive Divergence ####
- 🐑💬 Simply explain why scientists think about the positive and negative side effects of an event from action because there is some study explaining stability and possibility same as contrastive divergence. This contrastive divergence or approximate maximum likelihood of two sequences used in the network training method, starting from similarity and learning of events or input can create branches of action that can determine possibility example as the Markov chain distribution [Jump To]( https://www.researchgate.net/profile/Binu-Nair/publication/313556538 ) Possibility matrix [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#possibility-matrix )

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/contrastive-divergence.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/contrastive-divergence.jpg">
</picture>
<br>Ref[10]: Contrastive Divergence</br>
</p>
<br></br>

- 🐑💬 Markov chain is a method for determining the likelihood of two or more sequences with the possibility of input events and action. This method is famous because of the traceability and adjustable of matrix rules it does not require a large calculation process by itself but the methods they feedback value into their calculation matrix table. Scientists solved this problem in time by using the Eigant significant value to reduce the sizes of the calculation matrix. It allowed the likelihood of approximate processes performed on significant rows or areas [Jump To]( https://math.stackexchange.com/questions/3397763/calculating-probability-in-markov-chains )

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/markov-chain.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/markov-chain.jpg">
</picture>
<br>Ref[11]: Markov chain model</br>
</p>

#### Forward pass ####
- 🐑💬 Example of the forward network ( layer ) with probability statistics and distribution domain. The sigmoid activation function works as a condition filter allowing a change of value in scopes to pass the same as the bandpass-filter. The linear rectangular activation filter determines the contrast between two nodes equivalent [ 0, 0 ]. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#power-series )
```
X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], tf.float32)

v_state = X
print ("Input: ", v_state)

h_bias = tf.constant([0.1, 0.1])
print ("hb: ", h_bias)
print ("w: ", W)

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)  #probabilities of the hidden units
print ("p(h|v): ", h_prob)

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random.uniform(tf.shape(h_prob)))) #states
print ("h0 states:", h_state)
```

#### Results ####
```
Input:  tf.Tensor([[1. 0. 0. 1. 0. 0. 0.]], shape=(1, 7), dtype=float32)
hb:  tf.Tensor([0.1 0.1], shape=(2,), dtype=float32)
w:  tf.Tensor(
[[-1.7661804   0.4002081 ]
 [ 2.4241676   0.60449976]
 [-1.0282192  -0.63026726]
 [-2.3212638   0.8039171 ]
 [ 1.3049473   1.0474973 ]
 [ 0.4494173   0.47940415]
 [-0.64623755  0.6346767 ]], shape=(7, 2), dtype=float32)
p(h|v):  tf.Tensor([[0.01820932 0.7865284 ]], shape=(1, 2), dtype=float32)
h0 states: tf.Tensor([[0. 0.]], shape=(1, 2), dtype=float32)
```

#### Backward pass ####
- 🐑💬 The backward process can do both update weight or bias but there is a profitability from known state then the update state helps the learning process by accelerating the learning process. Some network layers require previous state sequence not only the output because of its benefit in learning. ( 👧💬 🎈 Learn from example is how our brain are training see the target and see the result then aim to the process ) *Image masking to find object from new sample example of learning from label [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#image-masking )   
```
vb = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print ("b: ", vb)
v_prob = tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + vb)
print ("p(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - tf.random.uniform(tf.shape(v_prob))))
print ("v probability states: ", v_state)
```

#### Probability distribution ####
- 🐑💬 I have an example of probability distribution as we can play AI Games with random functions and scopes by our learning pattern from the games, the same as in the experiment we added noise to input variables to prove our solution durability.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/flappy_distance.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/flappy_distance.jpg">
</picture>
</p>

#### Probability distribution implement ####
```
def random_action(  ): 
	
	gameState = p.getGameState()
	player_y_array = gameState['player_y']
	player_vel_array = gameState['player_vel']
	next_pipe_dist_to_player_array = gameState['next_pipe_dist_to_player']
	next_pipe_top_y_array = gameState['next_pipe_top_y']
	next_pipe_bottom_y_array = gameState['next_pipe_bottom_y']
	next_next_pipe_dist_to_player_array = gameState['next_next_pipe_dist_to_player']
	next_next_pipe_top_y_array = gameState['next_next_pipe_top_y']
	next_next_pipe_bottom_y_array = gameState['next_next_pipe_bottom_y']
	
	gap = (( next_pipe_bottom_y_array - next_pipe_top_y_array ) / 2 )
	top = next_pipe_top_y_array
	target = top + gap
	
	space = 512 - pipe_gap 
	upper_pipe_buttom = next_pipe_top_y_array + 0.8 * space
	
	coeff_01 = upper_pipe_buttom
	coeff_02 = 512 - player_y_array
	
	temp = tf.random.normal([2], 0.001, 0.5, tf.float32)
	# temp = tf.ones([2], tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02 ], shape=(2, 1), dtype=tf.float32))
	# temp = tf.nn.softmax(temp)
	
	temp = tf.math.argmax(temp)
	action = int(temp[0])
	
	action_name = list(actions.values())[action]
	action_name = [ x for ( x, y ) in actions.items() if y == action_name]
	
	print( "steps: " + str( step ).zfill(6) + " action: " + str(action_name) + " coeff_01: " 
          + str(int(coeff_01)).zfill(6) + " coeff_02: " 
          + str(int(coeff_02)).zfill(6) 

	)

	return action
```

#### Objective function ####
- 🐑💬 In the study materials if we understand the linear algorithm property we can use the summary function perform operation as a linear function with linear logarithms the example has  to explain about savings of the calculation process by the logarithm function. Look at the Eigant values method but now try to use the logarithms function in the small iterations it requires a lot of power because the results of the logarithms function are in small decimals numbers and the change of their values are very small but in larger scales, they are saved power of calculation because it updates on significant data learning and it does not required every time update as algorithm and differentiate the value of them are not significant. Weights response function [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#weight-response-calculation )

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/objective_function.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/objective_function.jpg">
</picture>
</p>

- 🐑💬 The learning feature in study material is image transformation and you can safely communicate a message.

#### Communication and interaction logs in JSON format ####
```
data_string = json_format.MessageToJson(example)
example_binary = tf.io.decode_json_example(data_string)

example_phase = tf.io.parse_example(
serialized=[example_binary.numpy()],
features = { 	
                "1": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64),
                "2": tf.io.FixedLenFeature(shape=[ 183 * 275 * 3 ], dtype=tf.int64)
            })
```
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/JSON_image.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/JSON_image.jpg">
</picture>
</p>

* ML0120EN-Eager_Execution.ipynb
- 🐐💬 Eager execution is introduced in TF2.X to support the graph methodology and its improvement to support the backward algorithm by significant value estimation because of the similar nodes on the same iteration running should provide close value and this can estimate by Eigant significant values ```tf.executing_eagerly()```. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#re-enforcement-learning )

```
# enable eager execution mode or default in TF 2.X
enable_eager_execution()

# disable eager execution mode or default in TF 1.X
disable_eager_execution()
```

#### None-eagle session running ####
🐐💬 Create a session object and execute the thread this process is the same as the worker standard process or customizable. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#tf-variables-in-gradients-optimization )
```
# Define variables
a = tf.constant(np.array([1., 2., 3.]))
b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b, 1)

# Create session and running
session = tf.compat.v1.Session()
output = session.run(c)

# Close session
session.close()
```

#### Eagle session running ####
🐐💬 You can work with Tf.variable, TF.placeholder, or TF.constant with eagle mode but the module function with none-public variable performs accumulate function by TF.variable and TF.placeholder to use benefits from the performance of graph methodology. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#data-conditions )
```
# Define variables
a = tf.constant(np.array([1., 2., 3.]))
b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b,1)

# Working with constant variable
c.numpy()
```

#### Applied sciences for basics neurons networks and adaptation ####
* CNN.ipynb
- 🦭💬 Building convolution networks class, One of the outputs of the convolution networks can estimate of target size value by result of 2 * padding size + kernel size over stride matrix size. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#games-simulation )    

```
# Create the model object using CNN class

model = CNN(out_1=16, out_2=32)

# Train the model

# Number of times we want to train on the taining dataset
n_epochs=3
# List to keep track of cost and accuracy
cost_list=[]
accuracy_list=[]
# Size of the validation dataset
N_test=len(validation_dataset)

# Model Training Function
def train_model(n_epochs):
    # Loops for each epoch
    for epoch in range(n_epochs):
        # Keeps track of cost for each epoch
        COST=0
        # For each batch in train loader
        for x, y in train_loader:
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction based on X value
            z = model(x)
            # Measures the loss between prediction and acutal Y value
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Cumulates loss 
            COST+=loss.data
        
        # Saves cost of training data of epoch
        cost_list.append(COST)
        # Keeps track of correct predictions
        correct=0
        # Perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            # Makes a prediction
            z = model(x_test)
            # The class with the max value is the one we are predicting
            _, yhat = torch.max(z.data, 1)
            # Checks if the prediction matches the actual value
            correct += (yhat == y_test).sum().item()
        
        # Calcualtes accuracy and saves it
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
     
train_model(n_epochs)
```

#### Iterate Pytorch data loader and plot  ####
```
# Plot samples

count = 0
for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
    z = model(x)
    _, yhat = torch.max(z, 1)
    # if yhat != y:
    show_data((x, y))
    plt.show()
    print("yhat: ",yhat)
    count += 1

    if count >= 5:
        break  
```

* Neural_Network_ReLU_vs_Sigmoid.ipynb
- 🐐💬 Rectified Linear Unit (ReLU) and Sigmoid activation functions are linear unit conditions approach for input to output as logits and linear exponential ( 🐑💬 It is a linear function scaled by the logarithm as ratios )

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid.jpg">
</picture>
<br>Ref[12]: Sigmoid function example from Geeks for Geeks.</br>
</p>
🐐💬 Slowly change its output value until approx [𝝅 / 2 to 𝝅 / 4] creating significant value that can be determined with requirements.

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/RELU.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/RELU.jpg">
</picture>
<br>Ref[14]: ReLU function example from Geeks for Geeks.</br>
</p>	

🐐💬 Rectify unit angle to identify output from target ratios. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#tf-variables-in-gradients-optimization )

```
# Create variable
x = tf.constant(np.array([1., 2., 3.]))

# Create a linear function object from torch.nn.Linear
linear_function_1 = torch.nn.Linear(num_in, num_out)

# Create a linear function object from torch.nn.Linear
linear_function_2 = torch.nn.Linear(num_in, num_out)

# Running a Sigmoid function object from torch.sigmoid
sigmoid_function_result = torch.sigmoid(linear_function_1(x))

# Running a ReLU function object from torch.relu
relu_function_result = torch.relu(linear_function_2(x))
```

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid_relu_loss.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid_relu_loss.jpg">
</picture>
<br>Ref[15]: Sigmoid and ReLU training loss value iterations</br>
</p>

🐐💬 Rectangular unit is growing faster and steadily for natural function as their behavior when Sigmoid function has a large learning area slower grow is benefit when learning of specific tasks or pattern because it memorizes benefits.

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid_relu_accuracy.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sigmoid_relu_accuracy.jpg">
</picture>
<br>Ref[16]: Sigmoid and ReLU training accuray value iterations</br>
</p>

🐐💬 Non-complex or non-natural function, the rectangular unit is just right to identify of input pattern within small iterations but longer and more complex require memory for learning. Further the ```nn.CrossEntropyLoss()``` is clearly can identify as red and white colours. There are benefits of feedback ReLU against the environment and Sigmoid against real-complex problems this does not mean one is better than another but supporting of a rocket launcher can be built with Sigmoid but lunch with ReLU. ( 🦭💬 Sometimes performance also included clearly identify )

* Simple_Neural_Network_for_XOR.ipynb
- 🐐💬 Nuerons Networks with One Hidden Layer ( Noisy XOR ) [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#games-simulation )

#### Define Accuracy scores calculation by opposite rectangular of the square area ####
```
# Calculate the accuracy

def accuracy(model, data_set):
    # Rounds prediction to nearest integer 0 or 1
    # Checks if prediction matches the actual values and returns accuracy rate
    return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())
```

#### Hint for some idea with mean relative error ####
* My computer still not recovery but see my example and ideas [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#games-simulation ) 💻💩💩
```
m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
m.update_state([1, 3, 2, 3], [2, 4, 6, 8])
m.result().numpy()

>> precision 1.25
```
🦤💬 I think they also do not understand precision too 🧸💬 Only practice we know. 👧💬 🎈 When nothing is the same they are on target or out of map. 🐨🎁🎵🎶 Who⁉️
<br>Ref[17]: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanRelativeError</br>

🐐💬 Approximate one over three is the area of blue colour, this way we also can measure the network's performance learning too.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xor_loss_accuracy.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xor_loss_accuracy.jpg">
</picture>
<br>Ref[18]: XOR logic loss and accuracy value learning iterations </br>
</p>

🐐💬 They are alignment with SGD function and we can have a perpendicular SGD function for area scapes.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xor_learning.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xor_learning.jpg">
</picture>
<br>Ref[19]: XOR logic learning graph </br>
</p>

* Support_Vector_Machines_vs_Vanilla_Linear_Classifier.ipynb
- 🧸💬 Support Vector Machine and Vanilla Linear Classifier, in this exercise IBM talking about linear regression with output layer norminal domain. The exercise compared two domains in three examples. Support vector machine can mean vector images input learning machine for example the categorize problem in the example she performed SVG with gradient alignment lines but she found the secrete of singular value decompositions ```s, u, v = svd(a)```  ```🦭💬 A Hamm~ what is left singular vector, right singular vector and singular with shape⁉️ ``` Ref[18]

🧸💬 Inputs are unique and connect directly compare one point to one point same as our hair colours but regularization helps with this kind of problem by reducing some complex settings from the input that is not necessary with a specific process for possible operation and reducing none required repeating task. 👧💬 🎈 Simply explain as tolerance penalty games until acceptable rates.
```
logit = LogisticRegression(C=0.01, penalty='l1', solver='saga', tol=0.1, multi_class='multinomial')
```

#### LogisticRegression model running ####
🧸💬 Logit scores are the comparison of two lists which are x_test and y_test. Not recommended but this is a trick I give you to create un-supervised labels when you do not need to use a worksheet program or manual input by comparing of sample list with the target label list and training, you can find some secrete values are present by chance only for determine the networks can learn about all input categorize or they need to be more complex. 🎵🎶   🥺💬 Start I test with Piano notes music to see solution stability. ( re-mapping method ) [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#power-series )
```
# Training LogisticRegression model with data and label
logit.fit(X_train_logistic, y_train)

# Sample usage predicts Y value from input data X
y_pred_logistic = logit.predict(X_test_logistic)

# Evluation scores determination
print("Accuracy: "+str(logit.score(X_test_logistic, y_test)))

>>> Accuracy: 0.7638888888888888
```

🧸💬 Contrast table on the table countable comparable of units on the units mapping matrix allows estimates in detail of countable ranges.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/confusion_matrix.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/confusion_matrix.jpg">
</picture>
<br>Ref[20]: Confusion Matrix </br>
</p>

🧸💬 Mediant and errors are not escapes of statistics results, high errors may perform of best scouts but bad category. In L1 or global normalize is high tolerance while internal normalize is less tolerance than L1. High tolerance is cost-effective to develop when regression selects of good sample to develop this is shown by statistics and must be true. 👧💬 🎈 They are both global and normalized higher develop create better results but cannot create good statistics  0.85 > 0.80 with 15% variance better than 0.90 > 0.95 with 9% variance ⁉️ 🐐💬 Yes if it is not human this is a handwriting mapping logistics problem. [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#optimization-problem )     
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/logistic_svm.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/logistic_svm.jpg">
</picture>
<br>Ref[21]: SVM and logistic network results comparison </br>
</p>


* Training_a_Neural_Network_with_Momentum.ipynb
- 🐑💬 Training neuron networks with momentum, some discovery of the truth is that some solutions need to compare neighbors to reveal the true value because they are not global present but they are significant in local representation and that should be included in our scope of interest because they are on the same domain presenting of development area we focus and we can work with the same variable domain. See it as two local minima by momentum finding [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#power-series )  

#### Create sample model with SGD optimizer training ####
```
# Initialize a dictionary to contain the cost and accuracy
# 🧸💬 Same as the historical record from Tensorflow and TensorFlow-Lite training
Results = {"momentum 0": {"Loss": 0, "Accuracy:": 0}, "momentum 0.1": {"Loss": 0, "Accuracy:": 0}}

# Train a model with 1 hidden layer and 50 neurons

# Size of input layer is 2, hidden layer is 50, and output layer is 3
# Our X values are x and y coordinates and this problem has 3 classes
Layers = [2, 50, 3]
# Create a model
model = Net(Layers)
learning_rate = 0.10
# Create an optimizer that updates model parameters using the learning rate, gradient, and no momentum
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Create a Data Loader for the training data with a batch size of 20
train_loader = DataLoader(dataset=data_set, batch_size=20)
# We create a criterion which will measure loss
criterion = nn.CrossEntropyLoss()
# Use the training function to train the model for 100 epochs
Results["momentum 0"] = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
# Prints the dataset and decision boundaries
plot_decision_regions_3class(model, data_set)
```

🐑💬 Some local values need to be updated by comparing local maxima for an efficient method.
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_learning.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_learning.jpg">
</picture>
<br>Ref[22]: SGD learning method </br>
</p>

🐑💬 The momentum method creates new updates of local maxima for performance. Polynomial function [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Polynomial-function )
<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_momentum.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_momentum.jpg">
</picture>
<br>Ref[23]: SGD with momentum=0.5 learning method </br>
</p>


* use-objectdetection-faster-r-cnn.ipynb
- 🦭💬 Object detection with faster R-CNN, recurrent convolution network is a unique network that has the property of power calculation sequence, input-output shape mapping, long-term memorization rates, and fast update significant when saving calculation power from convolution networks, arrays calculation and property of CNN networks. [Geoffrey Hinton]( https://www.utoronto.ca/news/ai-fuels-boom-innovation-investment-and-jobs-canada-report-says )

<p align="center" width="100%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/Geoffrey%20Hinton.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/Geoffrey%20Hinton.jpg">
</picture>
<br>Ref[23]: Geoffrey Hinton </br>
</p>

#### Use resnet50 to identify object identity ####
🦭💬 Object identity identification recurrent convolution layers performance on this task and they are support of more objects they can detect with criteria. 
Image region [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#Image-region ), Image masking [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#image-masking ) image-centre-detection [Jump To]( https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/README.md#imges-centre-detection )
```
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = transforms.Compose([transforms.ToTensor()])
img = transform(image)
pred = model([img])

pred[0]['labels']
>>> tensor([ 1, 15, 84,  2, 35, 84, 62,  2,  7, 84, 82, 84, 35, 84,  2, 35, 15, 42,
         2, 82, 62, 84, 62, 84,  7,  2, 84,  7,  2,  9, 84, 84,  2, 84,  2])

pred[0]['scores']
>>> tensor([0.9995, 0.3495, 0.2695, 0.2556, 0.2466, 0.1929, 0.1861, 0.1766, 0.1593,
        0.1528, 0.1484, 0.1392, 0.1295, 0.1290, 0.1249, 0.1208, 0.1094, 0.1026,
        0.1023, 0.1019, 0.0846, 0.0827, 0.0826, 0.0794, 0.0785, 0.0738, 0.0735,
        0.0713, 0.0669, 0.0622, 0.0595, 0.0578, 0.0575, 0.0553, 0.0520])

index=pred[0]['labels'][0].item()
COCO_INSTANCE_CATEGORY_NAMES[index]
>>> 'person'
```

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
|     3| https://www.xenonstack.com/blog/artificial-neural-network-applications | Famous types of Nuerons Networks |
|     4| https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification/mnist.py | MNIST image classification dataset |
|     5| https://www.tensorflow.org/text/tutorials/word2vec | Tensorflow word to vector method and solution |
|     6| https://www.tensorflow.org/api_docs/python/tf/keras/Model | Tensorflow and Keras sequential model and solution |
|     7| https://stackoverflow.com/users/7848579/jirayu-kaewprateep | DekDee contributed on StackOverflow |
|     8| https://www.ibm.com/products/watson-studio | IBM Watson Studio ( Free credits 200$ from Coursera course where I had free from IBM 1 month access ) |
|     9| https://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf | Restricted BOLTZMANN and collaborative filtering |
|    10| https://www.researchgate.net/profile/Binu-Nair/publication/313556538/figure/fig10/AS:460275318038537@1486749595664/Illustration-of-Contrastive-Divergence-for-training-RBMs.png | Contrastive Divergence |
|    11| https://math.stackexchange.com/questions/3397763/calculating-probability-in-markov-chains | Markov chain model |
|    12| https://www.geeksforgeeks.org/activation-functions | Activation function - Sigmoid |
|    13| https://www.tensorflow.org/api_docs/python/tf/math/sigmoid | Sigmoid function in TensorFlow |
|    14| https://www.geeksforgeeks.org/activation-functions | Activation function - ReLU |
|    15| IBM Neural_Network_RELU_vs_Sigmoid.ipynb | Sigmoid and ReLU training loss value iterations |
|    16| IBM Neural_Network_RELU_vs_Sigmoid.ipynb | Sigmoid and ReLU training accuray value iterations |
|    17| https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanRelativeError | Mean relative error for accuracy matric |
|    18| https://www.tensorflow.org/api_docs/python/tf/linalg/svd | Singular value decomposition |
|    19| IBM Support_Vector_Machines_vs_Vanilla_Linear_Classifier.ipynb | XOR logic learning graph |
|    20| https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/confusion_matrix.jpg | Confusion Matrix |
|    21| https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/logistic_svm.jpg | SVM and logistic network results comparison |
|    22| https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_learning.jpg | SGD learning method |
|    23| https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/sgd_momentum.jpg | SGD with momentum=0.5 learning method |
|    24| https://www.utoronto.ca/news/ai-fuels-boom-innovation-investment-and-jobs-canada-report-says | Geoffrey Hinton |

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

#### Imges centre detection ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/object_detection.jpg">
</picture>

#### Tensorflow and Pytorch Custom DataSet ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/custom_dataset.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/custom_dataset.jpg">
</picture>

#### DataLoader ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/data_loader.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/data_loader.jpg">
</picture>

#### Cylindical Problem ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/cylindical_problem.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/cylindical_problem.jpg">
</picture>

#### Weight response calculation ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/weight_response_cal.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/weight_response_cal.jpg">
</picture>

#### Convolution layer ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/convolution_layer.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/convolution_layer.jpg">
</picture>

#### Input identity identification method ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/Input_identity_identification.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/Input_identity_identification.jpg">
</picture>

#### Image pre-process and compressed commands ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/pre-process_compressed_commands.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/pre-process_compressed_commands.jpg">
</picture>

#### Multi-process and TF-agents ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/multi-process-TF-agents.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/multi-process-TF-agents.jpg">
</picture>

#### Data conditions ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/data_conditions.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/data_conditions.jpg">
</picture>

#### TF-variables in gradients optimization ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/gradients_optimization.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/gradients_optimization.jpg">
</picture>

#### Power Series ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/power_series.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/power_series.jpg">
</picture>

#### Remote execution sample ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/flappybird.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/flappybird.jpg">
</picture>

#### Customer Loss value estimation function ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/custom_lossfn.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/custom_lossfn.jpg">
</picture>

#### Internal function ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/internal%20function.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/internal%20function.jpg">
</picture>

#### Basics LSTM layer in object in image detection ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/LSTM-layer.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/LSTM-layer.jpg">
</picture>

#### Possibility matrix ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/possibility_matrix.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/possibility_matrix.jpg">
</picture>

#### Sentence words embedding - attention networks ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/attention-networks.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/attention-networks.jpg">
</picture>

#### Words or sequence confidence scores ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/confidence.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/confidence.jpg">
</picture>

#### Image masking ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/image_masking.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/image_masking.jpg">
</picture>

#### Re-enforcement learning ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/reinforcement_learning.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/reinforcement_learning.jpg">
</picture>

#### Optimization problem ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/optimization_problem.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/optimization_problem.jpg">
</picture>

#### Polynomial function ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/polynomial.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/polynomial.jpg">
</picture>

#### Image region ####
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/contour_image_region.jpg">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/contour_image_region.jpg">
</picture>
