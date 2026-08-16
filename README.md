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

This project extends the concepts learned from simple digit classification to a more challenging RGB image-classification  

##DAY 4 
# RNN and LSTM for IMDB Sentiment Classification

## Assignment

Load an NLP dataset, preprocess the text, build an RNN and an LSTM for
binary sentiment classification, evaluate both models, and
visualize/analyze the vanishing-gradient problem.

This project is designed for **Google Colab** and uses the IMDB
movie-review dataset.

## 1. Objectives

By completing this task, you should understand:

-   NLP sequence data
-   Tokenization and vocabulary
-   Token IDs and padding
-   Word embeddings
-   RNN hidden states
-   LSTM cell state and gates
-   Binary classification
-   Sigmoid and `BCEWithLogitsLoss`
-   Backpropagation through time
-   Vanishing gradients
-   RNN vs LSTM comparison

## 2. Dataset

We use the Hugging Face dataset:

`stanfordnlp/imdb`

Labels:

  Label   Meaning
  ------- ----------
  0       Negative
  1       Positive

Load it in Colab:

``` python
!pip install -U datasets huggingface_hub

from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
```

If `load_dataset("imdb")` gives an `HfUriError`, use
`load_dataset("stanfordnlp/imdb")`.

Inspect an example:

``` python
print(dataset["train"][0])
```

## 3. Google Colab Device

Run this before creating the models:

``` python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
```

If Colab has a GPU, the result should normally be `cuda`; otherwise it
will be `cpu`.

If the runtime is restarted, variables such as `device`, `vocab`,
`train_loader`, and the models must be recreated by running the notebook
cells again.

## 4. Overall Pipeline

``` text
Raw Review
    ↓
Lowercase / Tokenize
    ↓
Vocabulary
    ↓
Token IDs
    ↓
Truncation + Padding
    ↓
Embedding
    ↓
 ┌───────────────┐
 │               │
RNN             LSTM
 │               │
Hidden State   Hidden + Cell State
 │               │
Linear Layer   Linear Layer
 │               │
Logit           Logit
 │               │
Sigmoid         Sigmoid
 │               │
Positive/Negative
```

## 5. Text Preprocessing

The neural network cannot directly process words. Words are converted
into token IDs.

Example:

``` text
"I love this movie"
        ↓
[15, 42, 87, 231]
```

Create a vocabulary:

``` python
from collections import Counter

counter = Counter()

for example in dataset["train"]:
    words = example["text"].lower().split()
    counter.update(words)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.most_common(10000):
    vocab[word] = len(vocab)

print("Vocabulary size:", len(vocab))
```

Encode text:

``` python
def encode_text(text):
    words = text.lower().split()

    return [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]
```

Pad/truncate sequences:

``` python
MAX_LENGTH = 200

def preprocess_text(text):
    tokens = encode_text(text)
    tokens = tokens[:MAX_LENGTH]

    if len(tokens) < MAX_LENGTH:
        tokens += [vocab["<PAD>"]] * (
            MAX_LENGTH - len(tokens)
        )

    return tokens
```

## 6. PyTorch Dataset and DataLoader

``` python
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]

        tokens = preprocess_text(text)

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

train_dataset = IMDBDataset(dataset["train"])
test_dataset = IMDBDataset(dataset["test"])

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```

## 7. RNN Model

RNN processes a sequence while maintaining a hidden state:

``` text
word1 → h1
word2 → h2
word3 → h3
...
```

Model:

``` python
import torch.nn as nn

class RNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.rnn(x)

        final_hidden = hidden[-1]

        output = self.fc(final_hidden)

        return output.squeeze(1)
```

Create it:

``` python
rnn_model = RNNClassifier(
    vocab_size=len(vocab)
).to(device)

print(rnn_model)
```

## 8. Train RNN

For binary sentiment classification:

``` python
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    rnn_model.parameters(),
    lr=0.001
)
```

Training:

``` python
epochs = 3

rnn_train_losses = []

for epoch in range(epochs):

    rnn_model.train()
    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = rnn_model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    rnn_train_losses.append(average_loss)

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )
```

## 9. Evaluate RNN

``` python
def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total
```

``` python
rnn_accuracy = evaluate_model(
    rnn_model,
    test_loader
)

print(f"RNN Test Accuracy: {rnn_accuracy:.2f}%")
```

## 10. LSTM Model

LSTM means **Long Short-Term Memory**.

Compared with a basic RNN, LSTM uses:

-   Cell state
-   Forget gate
-   Input gate
-   Output gate

These mechanisms help the network preserve important information over
longer sequences.

``` python
class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]

        output = self.fc(final_hidden)

        return output.squeeze(1)
```

Create it:

``` python
lstm_model = LSTMClassifier(
    vocab_size=len(vocab)
).to(device)

print(lstm_model)
```

## 11. Train LSTM

``` python
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=0.001
)

lstm_train_losses = []

for epoch in range(epochs):

    lstm_model.train()
    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = lstm_model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    lstm_train_losses.append(average_loss)

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )
```

Evaluate:

``` python
lstm_accuracy = evaluate_model(
    lstm_model,
    test_loader
)

print(f"LSTM Test Accuracy: {lstm_accuracy:.2f}%")
```

## 12. Compare RNN and LSTM

``` python
print(f"RNN Accuracy:  {rnn_accuracy:.2f}%")
print(f"LSTM Accuracy: {lstm_accuracy:.2f}%")
```

Visualize:

``` python
models = ["RNN", "LSTM"]
accuracies = [rnn_accuracy, lstm_accuracy]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)

plt.ylabel("Accuracy (%)")
plt.title("RNN vs LSTM Test Accuracy")
plt.ylim(0, 100)

plt.show()
```

## 13. Vanishing Gradient

During training:

``` text
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradients
   ↓
Weight Updates
```

For a long RNN sequence:

``` text
h1 → h2 → h3 → ... → h100
```

the gradient must travel backward through many time steps.

If it is repeatedly multiplied by values smaller than 1, it can become
extremely small.

Example:

``` text
0.5^5  = 0.03125
0.5^10 = 0.0009765625
0.5^20 ≈ 0.000001
```

This is the **vanishing-gradient problem**.

## 14. Conceptual Vanishing-Gradient Visualization

``` python
steps = np.arange(1, 31)
gradient = 0.5 ** steps

plt.figure(figsize=(10, 5))

plt.plot(
    steps,
    gradient,
    marker="o"
)

plt.xlabel("Time Steps")
plt.ylabel("Gradient Magnitude")
plt.title("Vanishing Gradient Demonstration")

plt.grid()
plt.show()
```

This graph is a conceptual demonstration of how repeated multiplication
by values below 1 can make gradients approach zero.

## 15. Inspect Actual RNN Gradients

Run one forward/backward pass:

``` python
rnn_model.train()

inputs, labels = next(iter(train_loader))

inputs = inputs.to(device)
labels = labels.to(device)

rnn_model.zero_grad()

outputs = rnn_model(inputs)

loss = criterion(outputs, labels)

loss.backward()
```

Print gradient norms:

``` python
for name, parameter in rnn_model.named_parameters():

    if parameter.grad is not None:

        print(
            name,
            parameter.grad.norm().item()
        )
```

The recurrent parameter:

``` text
rnn.weight_hh_l0
```

is particularly relevant because it represents hidden-to-hidden
recurrent connections.

## 16. Inspect LSTM Gradients

``` python
lstm_model.train()

inputs, labels = next(iter(train_loader))

inputs = inputs.to(device)
labels = labels.to(device)

lstm_model.zero_grad()

outputs = lstm_model(inputs)

loss = criterion(outputs, labels)

loss.backward()
```

``` python
for name, parameter in lstm_model.named_parameters():

    if parameter.grad is not None:

        print(
            name,
            parameter.grad.norm().item()
        )
```

## 17. Important Interpretation

Do not conclude:

> "RNN always has vanishing gradients and LSTM never has them."

A better conclusion is:

> Basic RNNs are more susceptible to vanishing gradients when learning
> long-term dependencies. LSTM uses a cell state and gates to improve
> information and gradient flow across long sequences.

The actual gradient values depend on sequence length, initialization,
architecture, data, and training conditions.

## 18. Final Results to Report

Your final notebook/report should include:

1.  Dataset description
2.  Sample review and label
3.  Preprocessing steps
4.  Vocabulary size
5.  Sequence length
6.  RNN architecture
7.  RNN training loss
8.  RNN test accuracy
9.  LSTM architecture
10. LSTM training loss
11. LSTM test accuracy
12. RNN vs LSTM accuracy graph
13. Vanishing-gradient explanation
14. Gradient measurements
15. Vanishing-gradient visualization
16. Final comparison and conclusion

## 19. Final Conclusion Template

You can use this as the basis of your conclusion:

> In this assignment, the IMDB movie-review dataset was used for binary
> sentiment classification. The text was preprocessed by tokenization,
> vocabulary creation, numerical encoding, truncation, and padding. An
> embedding layer was then used to convert token IDs into dense vectors.
>
> Two recurrent models were implemented: a basic RNN and an LSTM. The
> RNN maintains a hidden state while processing the sequence, whereas
> the LSTM uses both a hidden state and a cell state controlled by
> gates.
>
> The experiment also investigated the vanishing-gradient problem.
> During backpropagation through long sequences, gradients in recurrent
> networks can become very small. This makes learning long-term
> dependencies difficult. LSTM was designed to reduce this problem
> through its gated memory mechanism.
>
> The final RNN and LSTM accuracy, training losses, and gradient
> measurements were compared to understand the practical differences
> between the two architectures.

## 20. Viva Questions

### What is an RNN?

An RNN is a neural network designed for sequential data. It maintains a
hidden state that carries information from previous time steps.

### What is an LSTM?

LSTM is a recurrent neural network architecture with a cell state and
gates designed to preserve useful information over longer sequences.

### What is a vanishing gradient?

It occurs when gradients become extremely small during backpropagation,
making earlier parts of a sequence difficult to learn.

### Why does an RNN suffer from vanishing gradients?

Because gradients are propagated through many recurrent time steps and
can repeatedly be multiplied by small values.

### Why does LSTM help?

The cell state and gates provide a mechanism for preserving and updating
information over longer sequences, which helps gradient and information
flow.

### Why use an embedding layer?

It converts token IDs into dense learned vectors that are more useful
for neural-network processing than arbitrary integer IDs.

### Why use Sigmoid?

The task is binary classification. Sigmoid converts one output logit
into a probability between 0 and 1.

### Why use BCEWithLogitsLoss?

It is appropriate for binary classification and combines the sigmoid
operation with binary cross-entropy in a numerically stable
implementation.

### Why not Softmax?

This implementation uses one output logit for binary classification.
Softmax is commonly used when the model produces multiple mutually
exclusive class logits.


problem.

