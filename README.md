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
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xenonstack-neural-network-architecture-3-1-1.webp">
  <img alt="My sample applications" src="https://github.com/jkaewprateep/Neuron-Networks-review/blob/main/images/xenonstack-neural-network-architecture-3-1-1.webp">
</picture>
<br>Ref[2]: https://www.xenonstack.com/blog/artificial-neural-network-applications</br>

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
- 🦭💬 Create a sequential model from a pre-trained model, one method to provide them the data sources without being exposed or trapped by network package identification is by the dataset but some bad man try to use it as a hacker step. Inside the dataset can contain multiple functions to manipulate and sort data when they are stored as sequences of binary numbers or encrypted messages. 🦭💬 We noticed with someone running the dataset or later running.

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
|     3| https://www.xenonstack.com/blog/artificial-neural-network-applications | Famous types of Nuerons Networks |
|     4| https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification/mnist.py | MNIST image classification dataset |

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
