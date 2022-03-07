# Study Note

## M1: Introduction and Basics
Motivation for NN:
- template assumption on pixel level is strong and we need some form of abstraction (features)
- image features
  - every pixel intensitives
  - feature extractor:
    - color histogram
    - histogram of oriented gradients
    - bag of visual words (cluster of similar pic patches)
- loss function:
  - general: loss function is to quantify classifer performance and improve your model by optimizing your loss funciton (minimze loss) by adjusting weighted matrix
  - SVM loss: (linear classifier)
    - maximum margin at decision boundary
    - add some penalty when it is not assigned with correct class, otherwise, the loss function is zero
- for classfication, we want to intrepet scores as probability : softmax function
- Other:
  - Adversarial learning: techniques to deal with following problems:
    - poisoning: manipulate the data before it is used for training
    - evasion: manipulate the model to make incorrect predictions
    - model stealing: manipulate the model to learn about the model or data
    - [ref](https://towardsdatascience.com/what-is-adversarial-machine-learning-dbe7110433d6)

## M2: Basics
### L3: Basics Part: Regularization and Optimization
### L4: Computational Graph and Backpropagation Part 1

## M3: Convolutional Neural Networks
### L5: History of and Introduction to Neural Networks
### L6: Convolutional Neural Networks
activation function:
- ![choice](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/choice%20of%20activation%20function.png)
MLP(Multi-layer neural network)
- can model arbitrary boolean functions
- can model arbirtrarily complex decision boundaries
- ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/MLP%20fit%20any%20decision%20boundaries.png)
- can model a circle at any location
  - ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/large%20number%20of%20neuron%20can%20fire%20circle%20at%20any%20location.png)
- can combine with any other cicle
  - ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/adding%20circles.png)
- **conclusion:one-hidden layer MLP can model any classification boundary** 
  -![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/one%20hidden%20layer%20can%20model%20any%20classification%20boundary.png)
image classification problem:
- first layer must capture relevant patterns
  - input layer: feature detectors and network is function over detectors
- higer level neuron compose complex templates
- shifted unseen new pic
  - conventional MLPs are sensitive to location, but often the location of a pattern is not important
    - shift invariance: scanning for patterns
      - instead of passing a whole image into a MLP, we can slide a smaller MLP over the image
      - we can use different MLPs (with different parameters) for every location or, we can share parameters across the spatial doamin(translation invariance)
how:
- overall plot:
  - ![overall plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/overall%20convolutional%20NN.png)
- convolution layer: ConvNet is a sequence of convolution layers interspersed with activation functions
  - for a image, we use a filter(the same full depth of the input volume) to filter with the image (slide over the image spatially)
    - repeats it several times(since many neuro in one hidden layers)
      - get an activation maps once after a filter slided over the image
        - we also can repeat it several times
          - ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/illstruation%20of%20ConvNet.png)
          - one filter generate one activation map
            - caculation of size of activation map (filtering with filter F):
              - ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/size%20of%20activation%20map.png)
              - common to zero pad the border: (pad with 1 pixel border): (N+2 - F)/stride + 1
          - remeber that after applying filter the activation maps shrink their size
            - too fast is not good
   - pooling layer:
    - makes the representations smaller and more manageable
    - operates over each activation map independently
      - ![illustration plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/pooling%20layer.png)
      - max polling
        - ![illustration plot for max pooling](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/max%20pooling.png)
  - fully connected layer (FC layer)
    - contains neurons that connect to the entire input volume, as in ordinary neural networks
- summary:
  - convNets stack contains: convolutional layers, pooling layers, fully connected layers
  - trend towrds smaller filters and deeper architectures
  - typical architecture:
    -![plot of summarized architecture](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/convNetS%20architecture.png)
  
## M4: Training Neural Networks
### L7: Training Part I: Activation, Initialization, Preproc, Dropout, Batch norm
### L8: Training Part II: Updates & Momentum, Augmentation, Transfer Learning

## M5: Architectures (Feb 21)
### L9: Inverse Classroom
### L10: Network Architectures: AlexNet, VGG, ResNet, U-Net, ...

## M6: Architectures (con't)(Feb 28)
### L11: Inverse Classroom

## M7: Sequence Modeling (Mar 07)
### L12: RNNs and LSTM

## M8: Unsupervised Learning

## M9: Break

## M10: Generative Models

## M11: Current Topics

## M12: Current Topics (con't)

## M13: Current Topics (con't)

## M14: Wrap up
