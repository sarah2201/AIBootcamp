#DAY1
# Spam and Ham Message Classification

This project implements a binary text classification system to classify messages as either Spam or Ham.

Three machine learning classification algorithms are applied:

1. Naive Bayes
2. Support Vector Machine (SVM)
3. Decision Tree

The performance of each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

TF-IDF is used to convert text messages into numerical features suitable for machine learning models.
# Image Classification Using Neural Network



## DAY2 
## 📌 Project Overview

This project demonstrates how to build a neural network for image classification using the **CIFAR-10 image dataset**.

The objective is to apply the concepts learned about neural networks, activation functions, Softmax, logits, loss functions, optimizers, and image preprocessing to a new image-classification problem.

The project is implemented using **PyTorch** and developed in **Google Colab**.

---

## 🎯 Objective

The main objective of this project is to:

* Load an image dataset.
* Explore and inspect the images.
* Apply proper preprocessing to the images.
* Convert images into tensors.
* Create DataLoaders for training and testing.
* Build a neural network for classification.
* Use ReLU activation in hidden layers.
* Produce raw logits from the output layer.
* Apply Softmax to obtain class probabilities.
* Train the neural network using a loss function and optimizer.
* Evaluate the model on unseen test images.
* Visualize predictions and class probabilities.

---

## 📊 Dataset

### CIFAR-10

CIFAR-10 is an image classification dataset containing **60,000 color images** divided into 10 classes.

The dataset contains:

* 50,000 training images
* 10,000 testing images
* Image size: 32 × 32 pixels
* Color channels: 3 RGB channels
* Number of classes: 10

### Classes

| Label | Class      |
| ----- | ---------- |
| 0     | Airplane   |
| 1     | Automobile |
| 2     | Bird       |
| 3     | Cat        |
| 4     | Deer       |
| 5     | Dog        |
| 6     | Frog       |
| 7     | Horse      |
| 8     | Ship       |
| 9     | Truck      |

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Google Colab

---

## 🔄 Project Workflow

The complete workflow is:

```text
CIFAR-10 Dataset
       ↓
Load Images
       ↓
Explore Dataset
       ↓
Image Preprocessing
       ↓
Convert Images to Tensors
       ↓
Create DataLoaders
       ↓
Neural Network
       ↓
Linear Layers
       ↓
ReLU Activation
       ↓
Output Layer
       ↓
Raw Logits
       ↓
Softmax
       ↓
Class Probabilities
       ↓
Prediction
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Optimizer
       ↓
Updated Weights
       ↓
Evaluation
```

---

## 🖼️ Image Preprocessing

The CIFAR-10 images are converted into PyTorch tensors using `ToTensor()`.

The original image pixel values are represented approximately in the range:

```text
0 - 255
```

After conversion to tensors, the values are scaled to:

```text
0 - 1
```

The images are then normalized using the CIFAR-10 RGB channel mean and standard deviation.

Normalization uses the formula:

```text
normalized_value = (value - mean) / standard_deviation
```

This preprocessing helps make the input data more suitable for neural-network training.

---

## 🧠 Neural Network Architecture

The neural network used in this project is a fully connected neural network.

```text
Input Image
32 × 32 × 3
     ↓
Flatten
     ↓
3072 values
     ↓
Linear Layer
3072 → 512
     ↓
ReLU
     ↓
Linear Layer
512 → 256
     ↓
ReLU
     ↓
Linear Layer
256 → 128
     ↓
ReLU
     ↓
Linear Layer
128 → 10
     ↓
Raw Logits
     ↓
Softmax
     ↓
10 Class Probabilities
```

The final layer contains 10 neurons because CIFAR-10 contains 10 classes.

---

## 🔥 ReLU Activation

ReLU is used in the hidden layers.

The ReLU function is:

```text
ReLU(x) = max(0, x)
```

It converts negative values to zero while keeping positive values.

ReLU helps the neural network learn non-linear relationships in the image data.

---

## 📈 Softmax

The final layer produces raw logits.

Softmax converts the logits into probabilities for the 10 classes.

The probabilities add up to approximately 1.

For example:

```text
Airplane    0.02
Automobile  0.01
Bird        0.03
Cat         0.05
Deer        0.02
Dog         0.10
Frog        0.04
Horse       0.03
Ship        0.05
Truck       0.65
```

The largest probability is for `Truck`, so the predicted class is Truck.

---

## ⚙️ Loss Function

The project uses:

```text
CrossEntropyLoss
```

This loss function is appropriate for multi-class classification.

The model produces raw logits, and `CrossEntropyLoss` internally handles the appropriate normalization for the classification loss.

---

## 🚀 Optimizer

The project uses the **Adam optimizer**.

```text
Optimizer: Adam
Learning Rate: 0.001
```

The optimizer updates the neural-network weights using the gradients calculated during backpropagation.

---

## 🏋️ Training

During training, the following process occurs:

```text
Input Images
     ↓
Forward Pass
     ↓
Predictions
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Gradient Calculation
     ↓
Adam Optimizer
     ↓
Weight Updates
```

This process is repeated for multiple batches and epochs.

---

## 📊 Evaluation

After training, the model is evaluated using the test dataset.

The test dataset contains images that were not used to train the model.

The main evaluation metric is:

```text
Classification Accuracy
```

Accuracy is calculated as:

```text
Correct Predictions
------------------- × 100
Total Predictions
```

---

## 📷 Visualization

The project visualizes:

* Sample CIFAR-10 images
* Predicted classes
* Actual classes
* Softmax probabilities
* Training loss
* Training accuracy

These visualizations help understand how the neural network performs.

---

## 💻 Running the Project

This project was created for **Google Colab**.

### Steps

1. Open Google Colab.
2. Create a new notebook.
3. Copy the project code into the notebook.
4. Enable GPU if available:

   * Runtime
   * Change runtime type
   * Select GPU
5. Run the notebook cells from top to bottom.
6. Observe the training results.
7. Evaluate the test accuracy.
8. View the prediction and probability visualizations.

---

## 📁 Project Structure

```text
image-classification/
│
├── image_classification.ipynb
├── README.md
└── data/
```

The CIFAR-10 dataset is downloaded automatically when the notebook is executed.

---

## ✅ Expected Learning Outcomes

After completing this project, I should be able to:

* Load an image dataset.
* Understand image dimensions and channels.
* Perform image preprocessing.
* Convert images into tensors.
* Create training and testing DataLoaders.
* Build a neural network using PyTorch.
* Understand Linear layers and weights.
* Apply ReLU activation.
* Understand raw logits.
* Apply Softmax to obtain probabilities.
* Select the predicted class.
* Calculate classification loss.
* Train a neural network using backpropagation.
* Use an optimizer to update weights.
* Evaluate a classification model.
* Visualize predictions and training performance.

---

## 📝 Conclusion

In this project, a neural network was created to classify images from the CIFAR-10 dataset.

The images were first preprocessed and converted into tensors. A fully connected neural network with ReLU activation was then created. The final layer produced 10 raw logits corresponding to the 10 CIFAR-10 classes.

Softmax was used to understand the class probabilities and determine the predicted class. CrossEntropyLoss was used for multi-class classification, while the Adam optimizer was used to update the model's weights during training.

This project extends the concepts learned from simple digit classification to a more challenging RGB image-classification problem.

