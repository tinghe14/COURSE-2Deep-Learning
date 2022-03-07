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
