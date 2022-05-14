# Study Note
Learning objectives:
- understand neural networks
- know the terminology of machine and deep learning
- comprehend the architectures mentioned in the lecture
- fearlessly design, build, train networks, and reason about pitfalls and design choices
- gain intuition

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
- for classfication, we want to intrepet scores as probability : softmax function $P(Y=k|X=xi) = (e^s)k / \sum (e^s)j$
    - multinomial logistic regression
    - softmax loss: -log P, log-likelihood of true class (logistic regression: sigmoid, softmax gives uou essentially a multi normal extension of the sigmoid function)
- Other:
  - Adversarial learning: techniques to deal with following problems:
    - poisoning: manipulate the data before it is used for training
    - evasion: manipulate the model to make incorrect predictions
    - model stealing: manipulate the model to learn about the model or data
    - [ref](https://towardsdatascience.com/what-is-adversarial-machine-learning-dbe7110433d6)

## M2: Basics
### L3: Basics Part: Regularization and Optimization

Regularization:
- simple way: 
  - L2 penalty: add penalty on weight magnitude
  - L1 penalty: encourage sparsity on solution

Double Descent:
- bias variance tradeoff (y-axis: error of model, x-axis: number of parameters/capacity of model)
- as we increase the model caplaxity beyond interpolation threshold point our test error actually decrease again
- You started with a model that doesn't give sensible predictions because all the weights that you have put there in the very beginning are random numbers. So essentially the capacity of the model zero because it is can't really do anything sensible. As you start optimizing , you update your parameters , and some of your parameters start to make sensible decision and essentially increase the capacity of the models as yo u iterate right at some point. 
  - key note want you to understand: not all of the parameters that you have in such a deep model later on will actually contribute to meaningful insurance, most of what you have there will essentially only contribute some noise in the background.

Optimization:

### L4: Computational Graph and Backpropagation Part 1

over-fitting: the performance reduced from training data to same level of testing data since training dataset over explain the variation 

dataset design:
- parameter optimized via training stage
- twick hyperparmater during validation stage such as using different architecture(add activation)which won't be able to learn during training stage
- show the preformance in test dataset(if directly use validation set, it will get a over optimistic estimates)
- ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/dataset%20design.png)

SVM loss:
- ![plot](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/SVM%20loss%20plot.png)
- ![example](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/SVM%20loss%20example.png)
Softmax Loss:(multinomial logistic regression)
-![example](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/softmax%20loss%20example.png)
Summary from last lectures:
- quantify unhappiness with current model parameters:
  - SVM loss
  - softmax function and MLE:
    - such as: Kullback-Leibler divergence
  - two questions for today's lecture:
    - are good parameters unique?
    - how to get parameters that make us happy?

Regularization
- ![loss function with regulairzation term](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/loss%20function%20add%20regularization.png)
  - data loss
  - regularization: introduce preferences on weights
    - counteract overfitting by enforcing simpler models 抵消
    - aid optimization by shaping the loss function (adding curvature) 曲率
    - ![l1 l2 regularization](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/L1%20L2%20regularization.png)
    - more complex regularizers:
      - dropout
      - batch normalization
      - stochastic depth ...
hyperparameter: you can run a grid searchg overall possible hyperparameters, you simply evaluate your predictive performance and at the end you select for the one works well.

Optimization
- a target loss function in its full glory defining our preferred solution. How do we actually retrieve this solution?
  - the analytic approach: express derivatve, set to zero and find solutions
  - but it not possible in most cases since many local minimum points
    - gradient descent algorithm:
      - % while not_converged:
        - % gradient = eval_gradient(loss, data, weights)
        - % weights += - step_size * gradient (step_size is hyperparametrs)
          - too large step size can cause divergence
          - number of samples can be large
            - stochastic gradient descent: approximate sum over all samples by a sum over a much smaller minibatch
            - stochastic gradient descent algorithm:
            - % while not_converged:
              - % data_batch = sample_training_data(data, batch__size)
              - % gradient = eval_gradient(loss, data_batch, weights)
              - % weights += -step_size * gradient

How to make a comments on the error:
- dataset design: split data into train, validation, and test; hyperparameters chosen on validation, then evaluated on test

Computational Graphs:
- it helps us compute derivatives at arbitrary locations
  - complex expressions are broken down into easy functions
  - forward pass: evaluate expression
  - backward pass: recursive application of chain rule yeilds analytic gradients

Backpropagation
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

Reminder:
- Convolutional Neural Networks (ConvNets) stack: convolutional layers, pooling layers, fully connected layers
  - typical architecture: [(Conv -> Activation)*N -> Pool]*M -> (FC -> Activation)*F, Softmax
- Today will consider important for training: design choices + potential pitfalls

Connecting the dots:
- computational graphs: consecutively apply chain rule
- convolutional neural networks: same concept still applies + more complicated with shared weights
  - apply batched SGD: until not converged, do:
    - sample batch of data
    - forward prop through graph, compute loss
    - backward prop to compute gradients
    - update parameters using gradient

Activation
- [choice of activation function](https://github.com/tinghe14/COURSE-2Deep-Learning/blob/main/Plots%20in%20Study%20Notes/choice%20of%20activation%20function.png)
- sigmoid:
  - introduction: 
      - squashes input into [0,1]
      - saturating firing rate of a neuron
  - problems: 
    - gradient vanishes for saturated neurons
    - exp() computation is a bit expensive
    - outputs are not zero-centered [0,1]:
      - for here: what if all inputs are positive, gradients on w, show that local gradient is x -> all positive
        - which means gradient is either all positive or all negative
        - so we want zero-mean data
- tanh:
  - introduction: 
    - squashes input into [-1,1]
    - zero-centered output
  - problem:
    - gradient vanished for saturated neurons
 - ReLU: max(0,x)
  - introduction:
    - no saturation in positive regime
    - computationally efficient
    - converges much faster than previous functions
    - closer to biological neuron activation
  - problem:
    - not zero-centered
    - dead ReLU: will never actiave for some data (active ReLu: will activate for some of the data)
      - bias term to rescue: initialize with small psotive bias
- Leaky ReLU:
  - introduction:
    - no saturation
    - computationally efficient
    - converges much faster than previous
    - will not die
- exponential linear Unit
  - introduction:
    - benefits of ReLU
    - closer to zero mean
    - saturates in negative regime (noise-robust deactivation state)
  - problem:
    - exp() is a bit expensitive to compute
- summary:
  - start with ReLU
  - try leaky ReLU, PReLU, ELU, maybe even maxout  
  - try tanh if you have time
  - do not use sigmoid
  - be careful with learning rates

Initlization:
- where are we now:
  - architecture is decided (number of neurons, actiavtion functions)
  - close to start training
  - but where should we start, how do we initilize our weights/parameters
- weight initlization
  - Xavier initilization:  small random numbers with zero mean and well-defined standard deviation
    - works well but breaks with ReLU: because derivation is based on linear neuron assumption. After Xavier intialization, outputs will be in the linear regime for tanh and sigmoid but obviously not for ReLU,
  - He initilization: similar but a little difference in standard deviation
  - have a good initlization will speed up your work

Preprocessing:
- reminder of the sigmoid or ReLU problem: if all inputs are positibe, gradients on w shows that gradients is either all positive or all negative. this is ineffective updates
  - why this is a problem in image field? because the range of 'normal' image is [0,255]
  - solution:for images, mean centering(zero-center) can be sufficient (normalizaiton not necessary)
    - do not (necessarily) consider decorrelation, whitening or other techniques for images, but his may be different for other input data
    - attention: at inference time, apply the same transformation (eg mean substraction) with values extracted from the training data (always think about not leaking the information from your validation set)

Covariates shift and batch norm:
- covariate shifts: 
  - background: randomly sampling mini-batches: training assumes similar distribution
    - in practice (and although random), each mini-batch will have different distribution which cause covariate shift can happen in each layer
    - shifts can be large and can negatively affect training
      - we can eliminate covariate shift by 'moving' batches to zero mean and unit standard deviation
      - then, move entire collection to desirable location: batch normalization
        - rather than pre-conditioning data and hoping that nice properties are preserved, at each layer we re-condition during every forward pass
          - usually inserted right after fully connected or convolutional layers, right before activation
          - however, is unit gaussian activation necessarily what we want?
            - consider tanh or sigmoid activation:
              - batch normalization will limit the activation to the linear regim of these activation functions
                - in such case, negatively affects performance
                - there are other cases where you also would not want batch normalization, eg when magnitude matters
        - benefits of batch normalization:
          - improves gradient flow through networks and allows for higher learning rates
          - reduces strong dependence on initlization
          -  acts as regularization
        - what to do at testing time:
          - compute average mean and standard deviation across multiple batches, the nsave these values for inference 
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
