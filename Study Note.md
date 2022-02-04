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

## M3: Convolutional Neural Networks

## M4: Training Neural Networks

## M5: Architectures

## M6: Architectures (con't)

## M7: Sequence Modeling

## M8: Unsupervised Learning

## M9: Break

## M10: Generative Models

## M11: Current Topics

## M12: Current Topics (con't)

## M13: Current Topics (con't)

## M14: Wrap up
