# %% [markdown]
# # Introduction to pytorch models
#
# ## Introduction to autodiff
#
# Load needed libraries
# $\newcommand\p[1]{{\left(#1\right)}}$
# $\newcommand\code[1]{\texttt{#1}}$

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# %% [markdown]
# Here is a simple example of how to find the minimum of the function
# $x\mapsto\p{x-3}^2$ using the autodiff functionality of Pytorch.
#
# First initialize a tensor `x` and indicate that we want to store a
# gradient on it.

# %%
x = torch.tensor([1.0], requires_grad=True)

# %% [markdown]
# Create an optimizer on parameters. Here we want to optimize w.r.t.
# variable `x`:

# %%
optimizer = optim.SGD([x], lr=0.01)

# %% [markdown]
# Create a computational graph using parameters (here only `x`) and
# potentially other tensors.

# Here we only want to compute $\p{x-3}^2$ so we define:
# %%
y = (x - 3) ** 2

# %% [markdown]
# Back-propagating gradients for `y` down to `x`. Don't forget to
# reset gradients before.

# %%
optimizer.zero_grad()
y.backward()

# %% [markdown]
# Use gradient on `x` to apply a one-step gradient descent.

# %%
optimizer.step()
x.grad
x

# %% [markdown]
# And last we iterate the whole process

# %%
it = 0
while it < 1000:
    loss = (x - 3) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 20 == 0:
        print('Iteration: %d, x: %f, loss: %f' % (it, x.item(), loss.item()))
    it += 1


# %% [markdown]
# ## Differentiate the exponential
#
# The exponential function can be approximated using its Taylor
# expansion:
# \\[
# \exp\p{z}\approx\sum_{k=0}^{N}\frac{z^k}{k!}
# \\]

# First define `x`, the "parameter" and build a computational graph from
# it to compute the exponential.
# %%
...

# %% [markdown]
# Compute the gradient and verify that it is correct

# %%
...


# %% [markdown]
# ## Solving equations with Pytorch
#
# Suppose we want to solve the following system of two equations
# \\[
# e^{-e^{-(x_1 + x_2)}} = x_2 (1 + x_1^2)
# \\]
# \\[
# x_1 \cos(x_2) + x_2 \sin(x_1) = 1/2
# \\]
#
# Find a loss whose optimization leads to a solution of the system of
# equations above.

# %%
# Define two functions
...


# %% [markdown]
# Use Pytorch autodiff to solve the system of equations

# %%
...


# %% [markdown]
# ## Linear least squares Pytorch implementation
#
# Every model in Pytorch is implemented as a class that derives from
# `nn.Module`. The two main methods to implement are:
#
# - `__init__`: Declare needed building blocks to implement forward pass
# - `forward`: Implement the forward pass from the input given as
#   argument

# %%
import torch.nn as nn

class LinearLeastSquare(nn.Module):
    def __init__(self, input_size):
        super(LinearLeastSquare, self).__init__()

        # Declaring neural networks building blocks. Here we only need
        # a linear transform.
        self.linear = ...

    def forward(self, input):
        # Implementing forward pass. Return corresponding output for
        # this neural network.
        return ...


# %% [markdown]
# ## Synthetic data
#
# We use the following linear model:
#
# \\[
# y = \langle\beta,x\rangle+\varepsilon
# \\]
#
# where \\(x\in\mathcal R^p\\) and \\(\varepsilon\sim\mathcal N(0, \sigma^2)\\).

# %%
import math
p = 512
N = 50000
X = torch.randn(N, p)
beta = torch.randn(p, 1) / math.sqrt(p)
y = torch.mm(X, beta) + 0.5 * torch.randn(N, 1)

# %% [markdown]
# ## Preparing dataset to feed Pytorch model

# %%
from torch.utils.data import TensorDataset

# Gather data coming from Pytorch tensors using `TensorDataset`
dataset = ...

# %%
from torch.utils.data import DataLoader
# Define `train_loader` that is an iterable on mini-batches using
# `DataLoader`
batch_size = ...
train_loader = ...

# %%
# Loss function to use
from torch.nn import MSELoss
loss_fn = ...

# %%
# Optimization algorithm
from torch.optim import SGD

# Instantiate model with `LinearLeastSquare` with the correct input
# size.
model = ...

# %%
# Use the stochastic gradient descent algorithm with a learning rate of
# 0.01 and a momentum of 0.9.
optimizer = ...

# %% [markdown]
# ## Learning loop

# %%
epochs = 10
losses = []
for i in range(epochs):
    for src, tgt in train_loader:
        # Forward pass
        ...

        # Backpropagation on loss
        ...

        # Gradient descent step
        ...

        losses.append(loss.item())

    print(f"Epoch {i}/{epochs}: Last loss: {loss}")


# %%
x = np.arange(len(losses)) / len(losses) * epochs
plt.plot(x, losses)

# %% [markdown]
# From the model what should be the minimum MSE?
# Noise distribution is \\(\varepsilon\sim\matcal\p{0, 0.25}\\) so the
# minimum MSE should be \\(\mathcal E\p{\epsilon^2}=0.25\\).
...

# %% [markdown]
# ## Learning loop with scheduler
#
# From convex optimization theory the learning rate should be decreasing
# toward 0. To have something approaching we use a scheduler that is
# updating the learning rate every epoch.

# %%
from torch.optim.lr_scheduler import MultiStepLR

# Define a scheduler
model = ...
optimizer = ...
scheduler = ...


# %%
# Implement the learning loop with a scheduler
...


# %% [markdown]
# ## Multi-layer perceptron
#
# Implement a multi-layer perceptron described by the following
# function:
# \\[
# f\p{x,\beta}=W_3\sigma\p{W_2\sigma{W_1 x}}
# \\]
# where \\(\sigma\p{x}=\max\p{x, 0}\\)

# %%
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MultiLayerPerceptron, self).__init__()

        # Define hyperparameters of neural network and building blocks
        ...

    def forward(self, x):
        # Implement forward pass
        ...

# %% [markdown]
# ## Synthetic 2-dimensional spiral dataset

# %%
n_classes = 3
n_loops = 2
n_samples = 1500

def spirals(n_classes=3, n_samples=1500, n_loops=2):
    klass = np.random.choice(n_classes, n_samples)
    radius = np.random.rand(n_samples)
    theta = klass * 2 * math.pi / n_classes + radius * 2 * math.pi * n_loops
    radius = radius + 0.05 * np.random.randn(n_samples)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta))).astype("float32"), klass

X_, y_ = spirals(n_samples=n_samples, n_classes=n_classes, n_loops=n_loops)
plt.scatter(X_[:, 0], X_[:, 1], c=y_)

# %%
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

X = torch.from_numpy(X_)
y = torch.from_numpy(y_)
dataset = TensorDataset(X, y)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loss_fn = CrossEntropyLoss()

# %%
X_, y_ = spirals(n_samples=1000, n_classes=n_classes, n_loops=n_loops)
X = torch.from_numpy(X_)
y = torch.from_numpy(y_)
test_set = TensorDataset(X, y)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# %%
model = MultiLayerPerceptron(2, 20, 20, n_classes)
optimizer = SGD(model.parameters(), lr=0.05)
# optimizer = Adam(model.parameters())

# %%
import copy

epochs = 1000
losses = []
models = []
for i in range(epochs):
    for src, tgt in train_loader:
        ...

    # Accuracy on the test set
    acc = 0.
    for src, tgt in test_loader:
        prd = model(src).detach().argmax(dim=1)
        acc += sum(prd == tgt).item()

    acc /= len(test_set)
    print(f"Epoch {i}/{epochs}: Test accuracy: {acc}")

    models.append(copy.deepcopy(model))


# %%
def get_image_data(model, colors, xs, ys):
    """Return color image of size H*W*4."""

    # Generate points in grid
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel())).astype("float32")
    points = torch.from_numpy(points)

    # Predict class probability on points
    prd = model(points).detach()
    prd = torch.nn.functional.softmax(prd, dim=1)

    # Build a color image from colors
    colors = torch.from_numpy(colors)
    img = torch.mm(prd, colors).numpy()
    img = img.reshape((ynum, xnum, 4))
    img = np.minimum(img, 1)

    return img

fig, ax = plt.subplots()

# Get n_classes colors in RGBa form
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import matplotlib as mpl
colors = mpl.colors.to_rgba_array(colors)[:n_classes, :4].astype("float32")

# Draw scatter plot of test set using colors
ax.scatter(X[:, 0], X[:, 1], c=colors[y])
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xnum, ynum = (int(i) for i in fig.dpi * fig.get_size_inches())

# Create discretization
xs = np.linspace(xmin, xmax, xnum)
ys = np.linspace(ymin, ymax, ynum)
img = get_image_data(model, colors, xs, ys)

ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=.7)


# %% [markdown]
# # Neural networks from scratch
#
# ## Libraries and dataset

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
X, y_ = make_moons(1000, noise=.2)
y = y_[:, np.newaxis]           # Make it a one-column matrix

# %% [markdown]
# Visualize the dataset

# %%
# <answer>
plt.scatter(*X.T, c=y_)
plt.show()
# </answer>

# %% [markdown]
# ## Activation functions
#
# ReLU and sigmoid function and their derivative (should work for numpy
# array of any dimension (1D, 2D,...))

# %%
def relu(v):
    # <answer>
    vv = np.copy(v)
    vv[vv < 0] = 0
    return vv
    # </answer>


def drelu(v):
    # <answer>
    vv = np.copy(v)
    vv[vv <= 0] = 0
    vv[vv > 0] = 1
    return vv
    # </answer>


def sigmoid(v):
    # <answer>
    return 1 / (1 + np.exp(-v))
    # </answer>


def dsigmoid(v):
    # <answer>
    return sigmoid(v) * (1 - sigmoid(v))
    # </answer>


# %% [markdown]
# ## Defining the neural network
#
# First define the shape of the neural network:
#
# - `n0`: size of input,
# - `n1`: size of hidden layer,
# - `n2`: size of output.

# %%
# <answer>
n0 = 2
n1 = 50
n2 = 1
# </answer>


# %% [markdown]
# Variables for weights, biases of each layers and intermediate
# variables to compute the gradient.

# %%
# Weights
W1 = np.random.randn(n0, n1)
W2 = np.random.randn(n1, n2)

# Biases
b1 = np.random.randn(n1)
b2 = np.random.randn(n2)

# Partial derivatives of output w.r.t. activations, see slide
# "Backpropagation equations"
Xx_1 = np.zeros((n2, n1))
Xx_2 = np.zeros((n2, n2))

# Partial derivatives of output w.r.t. biases, see slide
# "Backpropagation equations"
Xb_1 = np.zeros((n2, n1))
Xb_2 = np.zeros((n2, n2))

# Partial derivatives of output w.r.t. weigths, see slide
# "Backpropagation equations"
Xw_1 = np.zeros((n2, n1, n0))
Xw_2 = np.zeros((n2, n2, n1))

# Partial derivatives of loss w.r.t. weigths and biases, see slide
# "Cross entropy loss"
Lw_1 = np.zeros((n1, n0))
Lw_2 = np.zeros((n2, n1))
Lb_1 = np.zeros(n1)
Lb_2 = np.zeros(n2)

# %% [markdown]
# Define the learning rate and the activation functions along their
# derivatives at each layer:
#
# - `eta`: learning rate
# - `af1`, `daf1`: activation function and its derivative for hidden layer
# - `af2`, `daf2`: activation function and its derivative for output layer

# %%
# Define eta, af1, daf1, af2, daf2
# <answer>
eta = 0.01
af1 = relu
daf1 = drelu
af2 = sigmoid
daf2 = dsigmoid
# </answer>

# %% [markdown]
# ## The learning loop

# %%
nepochs = 15
for epoch in range(nepochs + 1):
    for idx, (x0, y2) in enumerate(zip(X, y)):
        # Implement the forward pass: use `W1`, `x0`, `b1`, `af1`, `W2`,
        # `x1`, `b2`, `af2` to define `z1`, `x1`, `z2`, `x2`.
        # <answer>
        z1 = W1.T @ x0 + b1
        x1 = af1(z1)
        z2 = W2.T @ x1 + b2
        x2 = af2(z2)
        # </answer>

        if idx % 100 == 0:
            print(f"Epoch: {epoch}, sample: {idx}, class: {y2}, prob: {x2}")

        # To initialize the recurrent relation (3), see slide
        # "Backpropagation equations"
        # <answer>
        Xx_2 = np.eye(n2)
        # </answer>

        # Update partial derivatives of output w.r.t. weights and
        # biases on second layer
        for i in range(n2):
            for p in range(n2):
                # See equation (2) in slide "Backpropagation equations"
                # <answer>
                Xb_2[i, p] = Xx_2[i, p] * daf2(z2[p])
                # </answer>
                for q in range(n1):
                    # See equation (1) in slide "Backpropagation equations"
                    # <answer>
                    Xw_2[i, p, q] = Xx_2[i, p] * x1[q] * daf2(z2[p])
                    # </answer>

        # Update partial derivatives of output w.r.t. activations
        for i in range(n2):
            for p in range(n1):
                Xx_1[i, p] = 0
                for j in range(n2):
                    # See equation (3) in slide "Backpropagation equations"
                    # <answer>
                    Xx_1[i, p] += Xx_2[i, j] * W2[p, j] * daf2(z2[j])
                    # </answer>

        # Update partial derivatives of output w.r.t. weights and
        # biases on first layer
        for i in range(n2):
            for p in range(n1):
                # See equation (2) in slide "Backpropagation equations"
                # <answer>
                Xb_1[i, p] = Xx_1[i, p] * daf1(z1[p])
                # </answer>
                for q in range(n0):
                    # See equation (1) in slide "Backpropagation equations"
                    # <answer>
                    Xw_1[i, p, q] = Xx_1[i, p] * daf1(z1[p]) * x0[q]
                    # </answer>

        # Compute partial derivatives of the loss w.r.t weights and
        # biases. For simplicity, we will use the MSE loss instead of
        # the cross-entropy loss because it does not require an
        # additional softmax step.
        for p in range(n1):
            for q in range(n0):
                Lw_1[p, q] = 0
                for i in range(n2):
                    # Partial derivatives of loss w.r.t. weigths, see
                    # slide "MSE loss"
                    # <answer>
                    Lw_1[p, q] += 2 * (x2[i] - y2[i]) * Xw_1[i, p, q]
                    # Lw_1[p, q] += (x2[i] - y2[i]) / (x2[i] * (1 - x2[i])) * Xw_1[i, p, q]
                    # </answer>

        for p in range(n2):
            for q in range(n1):
                Lw_2[p, q] = 0
                for i in range(n2):
                    # Partial derivatives of loss w.r.t. weigths, see
                    # slide "MSE loss"
                    # <answer>
                    Lw_2[p, q] += 2 * (x2[i] - y2[i]) * Xw_2[i, p, q]
                    # Lw_2[p, q] += (x2[i] - y2[i]) / (x2[i] * (1 - x2[i])) * Xw_2[i, p, q]
                    # </answer>

        for p in range(n2):
            Lb_2[p] = 0
            for i in range(n2):
                # Partial derivatives of loss w.r.t. biases, see slide
                # "MSE loss"
                # <answer>
                Lb_2[p] += 2 * (x2[i] - y2[i]) * Xb_2[i, p]
                # Lb_2[p] += (x2[i] - y2[i]) / (x2[i] * (1 - x2[i])) * Xb_2[i, p]
                # </answer>

        for p in range(n1):
            Lb_1[p] = 0
            for i in range(n2):
                # Partial derivatives of loss w.r.t. biases, see slide
                # "MSE loss"
                # <answer>
                Lb_1[p] += 2 * (x2[i] - y2[i]) * Xb_1[i, p]
                # Lb_1[p] += (x2[i] - y2[i]) / (x2[i] * (1 - x2[i])) * Xb_1[i, p]
                # </answer>

        # Gradient descent step: use `eta`, `Lw_1` `Lw_2` `Lb_1` `Lb_2` to
        # update `W1`, `W2`, `b1`, `b2`.
        # <answer>
        W1 -= eta * Lw_1.T
        W2 -= eta * Lw_2.T
        b1 -= eta * Lb_1
        b2 -= eta * Lb_2
        # </answer>

# %% [markdown]
# ## Vizualization

# %%
num = 250
x = np.linspace(X[:, 0].min(), X[:, 0].max(), num)
y = np.linspace(X[:, 1].min(), X[:, 1].max(), num)
XX, YY = np.meshgrid(x, y)
points = np.c_[XX.ravel(), YY.ravel()]

z1 = W1.T @ points.T + b1[:, np.newaxis]
x1 = af1(z1)
z2 = W2.T @ x1 + b2[:, np.newaxis]
x2 = af2(z2)

C = x2.reshape(num, num)

plt.contourf(XX, YY, C, cmap=plt.cm.RdBu, alpha=.5)
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(*X.T, c=y_, cmap=cm_bright)

plt.show()


# %% [markdown]
# # Image Style Transfer Using Convolutional Neural Networks
#
# This notebook implements the algorithm found in [(Gatys
# 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

# %%
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils

from PIL import Image

imsize = 128
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

# Load `content_img` as a torch tensor of size 3 * `imsize` * `imsize`
image = Image.open("./data/images/dancing.jpg")
content_img = loader(image)

# Load `style_img` as a torch tensor of size 3 * `imsize` * `imsize`
image = Image.open("./data/images/mondrian.jpg")
style_img = loader(image)

# %% [markdown]

# ## Feature extraction with VGG19
# The next cell is a CNN based on VGG19 which extracts convolutional
# features specified by `modules_indexes`. It is used to compute the
# features of the content and style image. It is also used to
# reconstruct the target image by backpropagation.

# %%
class VGG19Features(nn.Module):
    def __init__(self, modules_indexes):
        super(VGG19Features, self).__init__()

        # VGG19 pretrained model in evaluation mode
        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()

        # Indexes of layers to remember
        self.modules_indexes = modules_indexes

    def forward(self, input):
        # Define a hardcoded `mean` and `std` of size 3 * 1 * 1
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        # First center and normalize `input` with `mean` and `std`
        input_norm = ...

        # Add a fake mini-batch dimension to `input_norm`
        input_norm = ...

        # Install hooks on specified modules to save their features
        features = []
        handles = []
        for module_index in self.modules_indexes:

            def hook(module, input, output):
                # `output` is of size (`batchsize` = 1) * `n_filters`
                # * `imsize` * `imsize`
                features.append(output)

            handle = self.vgg19.features[module_index].register_forward_hook(hook)
            handles.append(handle)

        # Forward propagate `input_norm`. This will trigger the hooks
        # set up above and populate `features`
        self.vgg19(input_norm)

        # Remove hooks
        [handle.remove() for handle in handles]

        # The output of our custom VGG19Features neural network is a
        # list of features of `input`
        return features


# %% [markdown]

# The next cell defines the convolutional layers we will use to
# capture the style and the content. Look at the paper to see what are
# those.

# %%

# Indexes of interesting features to extract

# Define `modules_indexes`
modules_indexes = ...

vgg19 = VGG19Features(modules_indexes)
content_features = [f.detach() for f in vgg19.forward(content_img)]

# %% [markdown]

# ## Style features as gram matrix of convolutional features

# The next cell computes the gram matrix of `input`. We first need to
# reshape `input` before computing the gram matrix.

# %%
def gram_matrix(input):
    batchsize, n_filters, width, height = input.size()

    # Reshape `input` into `n_filters` * `n_pixels`
    features = ...

    # Compute the inner products between filters in `G`
    G = ...

    # We `normalize` the values of the gram matrix by dividing by the
    # number of element in each feature maps.
    return G.div(n_filters * width * height)


style_gram_features = [gram_matrix(f.detach()) for f in vgg19.forward(style_img)]

target = content_img.clone().requires_grad_(True)

# %% [markdown]

# ## Optimizer

# Look at the paper to see what is the algorithm they are using.
# Remember that we are optimizing on a target image.

# %%

# Define `optimizer` to use L-BFGS algorithm to do gradient descent on
# `target`
optimizer = ...

# %% [markdown]

# ## The algorithm

# From the paper, there are two different losses. The style loss and the
# content loss.

# Define `style_weight` the trade-off parameter between style and
# content losses
style_weight = ...

# %%

for step in range(500):
    # To keep track of the losses in the closure
    losses = {}

    # Need to use a closure that computes the loss and gradients to allow the
    # optimizer to evaluate repeatedly at different locations
    def closure():
        optimizer.zero_grad()

        # First, forward propagate `target` through our VGG19Features neural
        # network and store its output as `target_features`
        target_features = ...

        # Define `content_loss` on the first layer only
        content_loss = ...

        style_loss = 0
        for target_feature, style_gram_feature in zip(target_features, style_gram_features):
            # Compute Gram matrix
            target_gram_feature = ...

            # Add current loss to `style_loss`
            style_loss += ...

        # Compute combined loss
        loss = ...

        # Store the losses
        losses["loss"] = loss.item()
        losses["style_loss"] = style_loss.item()
        losses["content_loss"] = content_loss.item()

        # Backward propagation and return loss
        ...

    # Gradient step : don't forget to pass the closure to the optimizer
    ...

    if step % 10 == 0:
        print("step {}:".format(step))
        print(
            "Style Loss: {:4f} Content Loss: {:4f} Overall: {:4f}".format(
                losses["style_loss"], losses["content_loss"], losses["loss"]
            )
        )
        img = target.clone().squeeze()
        img = img.clamp_(0, 1)
        utils.save_image(img, "output-{}.png".format(step))


# %% [markdown]
# # Skipgram model trained on "20000 lieues sous les mers"
#
# ## Needed libraries

# You will need the following new libraries:
# - `spacy` for tokenizing
# - `gensim` for cosine similarities (use `gensim>=4.0.0`)

# You will also need to download rules for tokenizing a french text.
# ```python
# python -m spacy download fr_core_news_sm
# ```

# %%
import numpy as np
import torch
from torch import nn
import torch.optim as optim

import spacy
from gensim.models.keyedvectors import KeyedVectors

# %%
spacy_fr = spacy.load("fr_core_news_sm")


# %% [markdown]
# ## Tokenizing the corpus

# %%
# Use a french tokenizer to Create a tokenizer for the french language
with open("data/20_000_lieues_sous_les_mers.txt", "r", encoding="utf-8") as f:
    document = spacy_fr.tokenizer(f.read())

# Define a filtered set of tokens by iterating on `document`. Define a
# subset of tokens that are
#
# - alphanumeric
# - in lower case
# <answer>
tokens = [
    tok.text.lower()
    for tok in document if tok.is_alpha or tok.is_digit
]
# </answer>

# Make a list of unique tokens and dictionary that maps tokens to
# their index in that list.
# <answer>
idx2tok = list(set(tokens))
tok2idx = {token: i for i, token in enumerate(idx2tok)}
# </answer>

# %% [markdown]
# ## The continuous bag of words model

# %%
class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Define an Embedding module (`nn.Embedding`) and a linear
        # transform (`nn.Linear`) without bias.
        # <answer>
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.U_transpose = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        # </answer>

    def forward(self, center):
        # Implements the forward pass of the skipgram model
        # `center` is of size `batch_size`

        # `e_i` is of size `batch_size` * `embedding_size`
        # <answer>
        e_i = self.embeddings(center)
        # </answer>

        # `UT_e_i` is of size `batch_size` * `vocab_size`
        # <answer>
        UT_e_i = self.U_transpose(e_i)
        # </answer>

        # <answer>
        return UT_e_i
        # </answer>


# Set the size of vocabulary and size of embedding
VOCAB_SIZE = len(idx2tok)
EMBEDDING_SIZE = 32

# Create a Continuous bag of words model
skipgram = Skipgram(VOCAB_SIZE, EMBEDDING_SIZE)

# Send to GPU if any
device = "cuda:0" if torch.cuda.is_available() else "cpu"
skipgram.to(device)

# %% [markdown]
# ## Preparing the data

# %%
# Generate n-grams for a given list of tokens, use yield, use window length of n-grams
def ngrams_iterator(token_list, ngrams):
    """Generates successive N-grams from a list of tokens."""

    for i in range(len(token_list) - ngrams + 1):
        idxs = [tok2idx[tok] for tok in token_list[i:i+ngrams]]

        # Get center element in `idxs`
        center = idxs.pop(ngrams // 2)

        # Yield the index of center word and indexes of context words
        # as a Numpy array (for Pytorch to automatically convert it to
        # a Tensor).
        yield center, np.array(idxs)


# Create center, context data
NGRAMS = 5
ngrams = list(ngrams_iterator(tokens, NGRAMS))

BATCH_SIZE = 512
data = torch.utils.data.DataLoader(ngrams, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## Learn Skipgram model

# %%
# Use the Adam algorithm on the parameters of `skipgram` with a learning
# rate of 0.01
# <answer>
optimizer = optim.Adam(skipgram.parameters(), lr=0.01)
# </answer>

# Use a cross-entropy loss from the `nn` submodule
# <answer>
ce_loss = nn.CrossEntropyLoss()
# </answer>

# %%
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for i, (center, context) in enumerate(data):
        center, context = center.to(device), context.to(device)

        # Reset the gradients of the computational graph
        # <answer>
        skipgram.zero_grad()
        # </answer>

        # Forward pass
        # <answer>
        UT_e_i = skipgram.forward(center)
        # </answer>

        # Define one-hot encoding for tokens in context. `one_hots` has the same
        # size as `UT_e_i` and is zero everywhere except at location
        # corresponding to `context`. You can use `torch.scatter`.
        # <answer>
        one_hots = torch.zeros_like(UT_e_i).scatter(1, context, 1/(NGRAMS-1))
        # </answer>

        # Compute loss between `UT_e_i` and `one_hots`
        # <answer>
        loss = ce_loss(UT_e_i, one_hots)
        # </answer>

        # Backward pass to compute gradients of each parameter
        # <answer>
        loss.backward()
        # </answer>

        # Gradient descent step according to the chosen optimizer
        # <answer>
        optimizer.step()
        # </answer>

        total_loss += loss.data

        if i % 20 == 0:
            loss_avg = float(total_loss / (i + 1))
            print(
                f"Epoch ({epoch}/{EPOCHS}), batch: ({i}/{len(data)}), loss: {loss_avg}"
            )

    # Print average loss after each epoch
    loss_avg = float(total_loss / len(data))
    print("{}/{} loss {:.2f}".format(epoch, EPOCHS, loss_avg))


# %% [markdown]
# ## Prediction functions

# Now that the skipgram model is learned we can give it a word and see what
# context the model predicts.

# %%
def predict_context_words(skipgram, center_word, k=4):
    """Predicts `k` best context words of `center_word` according to model `skipgram`"""

    # Get index of `center_word`
    center_word_idx = tok2idx[center_word]

    # Create a fake minibatch containing just `center_word_idx`. Make sure that
    # `fake_minibatch` is a Long tensor and don't forget to send it to device.
    # <answer>
    fake_minibatch = torch.LongTensor([center_word_idx]).unsqueeze(0).to(device)
    # </answer>

    # Forward propagate through the skipgram model
    # <answer>
    score_context = skipgram(fake_minibatch).squeeze()
    # </answer>

    # Retrieve top k-best indexes using `torch.topk`
    # <answer>
    _, best_idxs = torch.topk(score_context, k=k)
    # </answer>

    # Return actual tokens using `idx2tok`
    # <answer>
    return [idx2tok[idx] for idx in best_idxs]
    # </answer>

# %%
predict_context_words(skipgram, "mille")
predict_context_words(skipgram, "nemo")


# %% [markdown]
# ## Testing the embedding
#
# We use the library `gensim` to easily compute most similar words for
# the embedding we just learned.

# %%
m = KeyedVectors(vector_size=EMBEDDING_SIZE)
m.add_vectors(idx2tok, skipgram.embeddings.weight.detach().cpu().numpy())

# %% [markdown]
# You can now test most similar words for, for example "lieues",
# "mers", "professeur"... You can look at `words_decreasing_freq` to
# test most frequent tokens.

# %%
unique, freq = np.unique(tokens, return_counts=True)
idxs = freq.argsort()[::-1]
words_decreasing_freq = list(zip(unique[idxs], freq[idxs]))

# %%
# <answer>
m.most_similar("lieues")
m.most_similar("professeur")
m.most_similar("mers")
m.most_similar("noire")
m.most_similar("mètres")
m.most_similar("ma")
# </answer>


# %% [markdown]
# # Word embedding and RNN for sentiment analysis
#
# The goal of the following notebook is to predict whether a written
# critic about a movie is positive or negative. For that we will try
# three models. A simple linear model on the word embeddings, a
# recurrent neural network and a CNN.

# %%
from timeit import default_timer as timer
from typing import Iterable, List


import appdirs                  # Used to cache pretrained embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# %% [markdown]
# ## The IMDB dataset

# %%
torch_cache = appdirs.user_cache_dir("pytorch")
train_iter, test_iter = datasets.IMDB(root=torch_cache, split=("train", "test"))

import random

TRAIN_SET = list(train_iter)
TEST_SET = list(test_iter)
random.shuffle(TRAIN_SET)
random.shuffle(TEST_SET)

# %%
TRAIN_SET[0]

# %% [markdown]
# ## Global variables
#
# First let's define a few variables. `EMBEDDING_DIM` is the dimension
# of the vector space used to embed all the words of the vocabulary.
# `SEQ_LENGTH` is the maximum length of a sequence, `BATCH_SIZE` is
# the size of the batches used in stochastic optimization algorithms
# and `NUM_EPOCHS` the number of times we are going thought the entire
# training set during the training phase.

# %%
# <answer>
EMBEDDING_DIM = 8
SEQ_LENGTH = 64
BATCH_SIZE = 512
NUM_EPOCHS = 10
# </answer>

# %% [markdown]
# We first need a tokenizer that take a text a returns a list of
# tokens. There are many tokenizers available from other libraries.
# Here we use the one that comes with Pytorch.

# %%
tokenizer = get_tokenizer("basic_english")
tokenizer("All your base are belong to us")

# %% [markdown]
# ## Building the vocabulary
#
# Then we need to define the set of words that will be understood by
# the model: this is the vocabulary. We build it from the training
# set.

# %%
def yield_tokens(data_iter: Iterable) -> List[str]:
    for data_sample in data_iter:
        yield tokenizer(data_sample[1])


special_tokens = ["<unk>", "<pad>"]
vocab = build_vocab_from_iterator(
    yield_tokens(TRAIN_SET),
    min_freq=10,
    specials=special_tokens,
    special_first=True)
UNK_IDX, PAD_IDX = vocab.lookup_indices(special_tokens)
VOCAB_SIZE = len(vocab)

vocab['plenty']

# %% [markdown]

# To limit the number of tokens in the vocabulary, we specified
# `min_freq=10`: a token should be seen at least 10 times to be part
# of the vocabulary. Consequently some words in the training set (and
# in the test set) are not present in the vocabulary. We then need to
# set a default index.

# %%
# vocab['pouet']                  # Error
vocab.set_default_index(UNK_IDX)
vocab['pouet']

# %% [markdown]
# # Collate function
#
# The collate function maps raw samples coming from the dataset to
# padded tensors of numericalized tokens ready to be fed to the model.

# %%
def collate_fn(batch: List):
    def collate(text):
        """Turn a text into a tensor of integers."""

        tokens = tokenizer(text)[:SEQ_LENGTH]
        return torch.LongTensor(vocab(tokens))

    src_batch = [collate(text) for _, text in batch]

    # Pad list of tensors using `pad_sequence`
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    # </answer>

    # Turn 2 (positive review) and 1 (negative review) labels into 1 and 0
    # <answer>
    tgt_batch = torch.Tensor([label - 1 for label, _ in batch])
    # </answer>

    return src_batch, tgt_batch


print(f"Number of training examples: {len(TRAIN_SET)}")
print(f"Number of testing examples: {len(TEST_SET)}")

# %%
collate_fn([
    (1, "i am Groot")
])

# %% [markdown]
# ## Training a linear classifier with an embedding
#
# We first test a simple linear classifier on the word embeddings.


# %%
class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Define an embedding of `vocab_size` words into a vector space
        # of dimension `embedding_dim`.
        # <answer>
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # </answer>

        # Define a linear layer from dimension `seq_length` *
        # `embedding_dim` to 1.
        # <answer>
        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)
        # </answer>

    def forward(self, x):
        # `x` is of size `seq_length` * `batch_size`

        # Compute the embedding `embedded` of the batch `x`. `embedded` is
        # of size `batch_size` * `seq_length` * `embedding_dim`
        # <answer>
        embedded = self.embedding(x)
        # </answer>

        # Flatten the embedded words and feed it to the linear layer.
        # `flatten` is of size `batch_size` * (`seq_length` * `embedding_dim`)
        # <answer>
        flatten = embedded.view(-1, self.seq_length * self.embedding_dim)
        # </answer>

        # Apply the linear layer and return a squeezed version
        # `l1` is of size `batch_size`
        # <answer>
        return self.l1(flatten).squeeze()
        # </answer>


# %% [markdown]
# We need to implement an accuracy function to be used in the `Trainer`
# class (see below).


# %%
def accuracy(predictions, labels):
    # `predictions` and `labels` are both tensors of same length

    # Implement accuracy
    # <answer>
    return torch.sum((torch.sigmoid(predictions) > 0.5).float() == (labels > .5)).item() / len(
        predictions
    )
    # </answer>


assert accuracy(torch.Tensor([1, -2, 3]), torch.Tensor([1, 0, 1])) == 1
assert accuracy(torch.Tensor([1, -2, -3]), torch.Tensor([1, 0, 1])) == 2 / 3


# %% [markdown]
# Train and test functions

# %%
def train_epoch(model: nn.Module, optimizer: Optimizer):
    model.to(device)

    # Training mode
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoader(
        TRAIN_SET, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    matches = 0
    losses = 0
    for sequences, labels in train_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Implement a step of the algorithm:
        #
        # - set gradients to zero
        # - forward propagate examples in `batch`
        # - compute `loss` with chosen criterion
        # - back-propagate gradients
        # - gradient step
        # <answer>
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        # </answer>

        acc = accuracy(predictions, labels)

        matches += len(predictions) * acc

    return losses / len(TRAIN_SET), matches / len(TRAIN_SET)

# %%
def evaluate(model: nn.Module):
    model.to(device)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    val_dataloader = DataLoader(
        TEST_SET, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    losses = 0
    matches = 0
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        acc = accuracy(predictions, labels)
        matches += len(predictions) * acc
        losses += loss.item()

    return losses / len(TEST_SET), matches / len(TEST_SET)


# %%
def train(model, optimizer):
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss, train_acc = train_epoch(model, optimizer)
        end_time = timer()
        val_loss, val_acc = evaluate(model)
        print(
            f"Epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Train acc: {train_acc:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Val acc: {val_acc:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )

# %%
def predict_sentiment(model, sentence):
    "Predict sentiment of given sentence according to model"

    tensor, _ = collate_fn([("dummy", sentence)])
    prediction = model(tensor)
    pred = torch.sigmoid(prediction)
    return pred.item()


# %%
embedding_net = EmbeddingNet(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LENGTH)
print(sum(torch.numel(e) for e in embedding_net.parameters()))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

optimizer = Adam(embedding_net.parameters())
train(embedding_net, optimizer)


# # %% [markdown]
# # ## Training a linear classifier with a pretrained embedding
# #
# # Load a GloVe pretrained embedding instead

# Download GloVe word embedding
glove = torchtext.vocab.GloVe(name="6B", dim="100", cache=torch_cache)

# Get token embedding of our `vocab`
vocab_vectors = glove.get_vecs_by_tokens(vocab.get_itos())

# tot_transferred = 0
# for v in vocab_vectors:
#     if not v.equal(torch.zeros(100)):
#         tot_transferred += 1

# tot_transferred, len(vocab)


# %%
class GloVeEmbeddingNet(nn.Module):
    def __init__(self, seq_length, vocab_vectors, freeze=True):
        super().__init__()
        self.seq_length = seq_length

        # Define `embedding_dim` from vocabulary and the pretrained `embedding`.
        # <answer>
        self.embedding_dim = vocab_vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        # </answer>

        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)

    def forward(self, x):
        # `x` is of size batch_size * seq_length

        # `embedded` is of size batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)

        # `flatten` is of size batch_size * (seq_length * embedding_dim)
        flatten = embedded.view(-1, self.seq_length * self.embedding_dim)

        # L1 is of size batch_size
        return self.l1(flatten).squeeze()


glove_embedding_net1 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
print(sum(torch.numel(e) for e in glove_embedding_net1.parameters()))

optimizer = Adam(glove_embedding_net1.parameters())
train(glove_embedding_net1, optimizer)

# %% [markdown]
# ## Use pretrained embedding without fine-tuning

# Define model and freeze the embedding
# <answer>
glove_embedding_net1 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
# </answer>


# %% [markdown]
# ## Fine-tuning the pretrained embedding

# %%
# Define model and don't freeze embedding weights
# <answer>
glove_embedding_net2 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=False)
# </answer>

# %% [markdown]
# ## Recurrent neural network with frozen pretrained embedding

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_vectors, freeze=True):
        super(RNN, self).__init__()

        # Define pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)

        # Size of input `x_t` from `embedding`
        self.embedding_size = self.embedding.embedding_dim
        self.input_size = self.embedding_size

        # Size of hidden state `h_t`
        self.hidden_size = hidden_size

        # Define a GRU
        # <answer>
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        # </answer>

        # Linear layer on last hidden state
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        # `x` is of size `seq_length` * `batch_size` and `h0` is of size 1
        # * `batch_size` * `hidden_size`

        # Define first hidden state in not provided
        if h0 is None:
            # Get batch and define `h0` which is of size 1 *
            # `batch_size` * `hidden_size`
            # <answer>
            batch_size = x.size(1)
            h0 = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size).to(device)
            # </answer>

        # `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim`
        embedded = self.embedding(x)

        # Define `output` and `hidden` returned by GRU:
        #
        # - `output` is of size `seq_length` * `batch_size` * `embedding_dim`
        #   and gathers all the hidden states along the sequence.
        # - `hidden` is of size 1 * `batch_size` * `embedding_dim` and is the
        #   last hidden state.
        # <answer>
        output, hidden = self.gru(embedded, h0)
        # </answer>

        # Apply a linear layer on the last hidden state to have a
        # score tensor of size 1 * `batch_size` * 1, and return a
        # tensor of size `batch_size`.
        # <answer>
        return self.linear(hidden).squeeze()
        # </answer>


rnn = RNN(hidden_size=100, vocab_vectors=vocab_vectors)
print(sum(torch.numel(e) for e in rnn.parameters() if e.requires_grad))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnn.parameters()), lr=0.001)
train(rnn, optimizer)

# %% [markdown]
# ## CNN based text classification

# %%
class CNN(nn.Module):
    def __init__(self, vocab_vectors, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        self.embedding_dim = self.embedding.embedding_dim

        self.conv_0 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(3, self.embedding_dim)
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(4, self.embedding_dim)
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(5, self.embedding_dim)
        )
        self.linear = nn.Linear(3 * 100, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input `x` is of size `seq_length` * `batch_size`
        embedded = self.embedding(x)

        # The tensor `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim` and should be of size `batch_size` *
        # (`n_channels`=1) * `seq_length` * `embedding_dim` for the
        # convolutional layers. You can use `transpose` and `unsqueeze` to make
        # the transformation.
        # <answer>
        embedded = embedded.transpose(0, 1).unsqueeze(1)
        # </answer>

        # Tensor `embedded` is now of size `batch_size` * 1 *
        # `seq_length` * `embedding_dim` before convolution and should
        # be of size `batch_size` * (`out_channels` = 100) *
        # (`seq_length` - `kernel_size[0]` + 1) after convolution and
        # squeezing.
        # Implement the convolution layer
        # <answer>
        conved_0 = self.conv_0(embedded).squeeze(3)
        conved_1 = self.conv_1(embedded).squeeze(3)
        conved_2 = self.conv_2(embedded).squeeze(3)
        # </answer>

        # Non-linearity step, we use ReLU activation
        # <answer>
        conved_0_relu = F.relu(conved_0)
        conved_1_relu = F.relu(conved_1)
        conved_2_relu = F.relu(conved_2)
        # </answer>

        # Max-pooling layer: pooling along whole sequence
        # Implement max pooling
        # <answer>
        seq_len_0 = conved_0_relu.shape[2]
        pooled_0 = F.max_pool1d(conved_0_relu, kernel_size=seq_len_0).squeeze(2)

        seq_len_1 = conved_1_relu.shape[2]
        pooled_1 = F.max_pool1d(conved_1_relu, kernel_size=seq_len_1).squeeze(2)

        seq_len_2 = conved_2_relu.shape[2]
        pooled_2 = F.max_pool1d(conved_2_relu, kernel_size=seq_len_2).squeeze(2)
        # </answer>

        # Dropout on concatenated pooled features
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # Linear layer
        return self.linear(cat).squeeze()


# %%
cnn = CNN(vocab_vectors)
optimizer = optim.Adam(cnn.parameters())
train(cnn, optimizer)

# %% [markdown]
# ## Test function


# %% [markdown]
# # The transformer architecture
#
# ## Needed libraries

# %%
from collections.abc import Iterable
from timeit import default_timer as timer
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor, nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

# %% [markdown]
# # Dataset

# %%
from written_numbers_dataset import NumberDataset

# %% [markdown]
# ## Vocabulary
#
# We first build a vocabulary out of a list of iterators on tokens.
# Here the vocabulary is already known. To have a vocabulary object,
# we still use `build_vocab_from_iterator` with `[VOCAB]`.
#
# We will also need four different special tokens:
#
# - A token for unknown words
# - A padding token
# - A token indicating the beginning of a sequence
# - A token indicating the end of a sequence
#
# First we choose a dataset

# %%
# Define a training set and a test set for a dataset.
# Number of sequences generated for the training set
# <answer>
train_set = NumberDataset()
test_set = NumberDataset(n_numbers=1000)
# </answer>


# %%
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
vocab_src = build_vocab_from_iterator([train_set.vocab_src], specials=special_tokens)
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = vocab_src.lookup_indices(special_tokens)
vocab_tgt = build_vocab_from_iterator([train_set.vocab_tgt], specials=special_tokens)

# %% [markdown]
# You can test the `vocab` object by giving it a list of tokens.

# %%
# vocab([<tokens>,...])

# %% [markdown]
# ## Collate function
#
# The collate function is needed to convert a list of samples from their raw
# form to a Tensor that a Pytorch model can consume. There are two different
# tasks:
#
# - numericalizing the sequence: changing each token in its index in the
#   vocabulary using the `vocab` object defined earlier
# - pad sequence so that they have the same length, see [here][pad]
#
# [pad]: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

# %%
def collate_fn(batch: List):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:

        # Numericalize list of tokens using `vocab`.
        #
        # - Don't forget to add beginning of sequence and end of sequence tokens
        #   before numericalizing.
        #
        # - Use `torch.LongTensor` instead of `torch.Tensor` because the next
        #   step is an embedding that needs integers for its lookup table.
        # <answer>
        src_tensor = torch.LongTensor(vocab_src(["<bos>"] + src_sample + ["<eos>"]))
        tgt_tensor = torch.LongTensor(vocab_tgt(["<bos>"] + tgt_sample + ["<eos>"]))
        # </answer>

        # Append numericalized sequence to `src_batch` and `tgt_batch`
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    # Turn `src_batch` and `tgt_batch` that are lists of 1-dimensional
    # tensors of varying sizes into tensors with same size with
    # padding. Use `pad_sequence` with padding value to do so.
    #
    # Important notice: by default resulting tensors are of size
    # `max_seq_length` * `batch_size`; the mini-batch size is on the
    # *second dimension*.
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    # </answer>

    return src_batch, tgt_batch


# %% [markdown]
# ## Hyperparameters

# %%
torch.manual_seed(0)

# Size of source and target vocabulary
SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_tgt)

# Number of epochs
NUM_EPOCHS = 20

# Size of embeddings
EMB_SIZE = 128

# Number of heads for the multihead attention
NHEAD = 1

# Size of hidden layer of FFN
FFN_HID_DIM = 16

# Size of mini-batches
BATCH_SIZE = 1024

# Number of stacked encoder modules
NUM_ENCODER_LAYERS = 1

# Number of stacked decoder modules
NUM_DECODER_LAYERS = 1

# %% [markdown]
# ## Positional encoding

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Define Tk/2pi for even k between 0 and `emb_size`. Use
        # `torch.arange`.
        # <answer>
        Tk_over_2pi = 10000 ** (torch.arange(0, emb_size, 2) / emb_size)
        # </answer>

        # Define `t = 0, 1,..., maxlen-1`. Use `torch.arange`.
        # <answer>
        t = torch.arange(maxlen)
        # </answer>

        # Outer product between `t` and `1/Tk_over_2pi` to have a
        # matrix of size `maxlen` * `emb_size // 2`. Use
        # `torch.outer`.
        # <answer>
        outer = torch.outer(t, 1 / Tk_over_2pi)
        # </answer>

        pos_embedding = torch.empty((maxlen, emb_size))

        # Fill `pos_embedding` with either sine or cosine of `outer`.
        # <answer>
        pos_embedding[:, 0::2] = torch.sin(outer)
        pos_embedding[:, 1::2] = torch.cos(outer)
        # </answer>

        # Add fake mini-batch dimension to be able to use broadcasting
        # in `forward` method.
        pos_embedding = pos_embedding.unsqueeze(1)

        self.dropout = nn.Dropout(dropout)

        # Save `pos_embedding` when serializing the model even if it is not a
        # set of parameters
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        # `token_embedding` is of size `seq_length` * `batch_size` *
        # `embedding_size`. Use broadcasting to add the positional embedding
        # that is of size `seq_length` * 1 * `embedding_size`.
        # <answer>
        seq_length = token_embedding.size(0)
        positional_encoding = token_embedding + self.pos_embedding[:seq_length, :]
        # </answer>

        return self.dropout(positional_encoding)


# %% [markdown]
# ## Transformer model

# %%
class Seq2SeqTransformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            emb_size: int,
            nhead: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # Linear layer to compute a score for all tokens from output
        # of transformer
        # <answer>
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # </answer>

        # Embedding for source vocabulary
        # <answer>
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        # </answer>

        # Embedding for target vocabulary
        # <answer>
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        # </answer>

        # Positional encoding layer
        # <answer>
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        # </answer>

    def forward(
            self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            memory_key_padding_mask: Tensor,
    ):
        # Embed `src` and `trg` tensors and add positional embedding.
        # <answer>
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # </answer>

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # Use the encoder part of the transformer to encode `src`.
        # <answer>
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )
        # </answer>

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # Use the decoder par of the transformer to decode `tgt`
        # <answer>
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
        # </answer>

    def encode_and_attention(self, src: Tensor, src_mask: Tensor):
        """Used at test-time only to retrieve attention matrix."""

        src_pos = self.positional_encoding(self.src_tok_emb(src))
        self_attn = self.transformer.encoder.layers[-1].self_attn
        att = self_attn(src_pos, src_pos, src_pos, attn_mask=src_mask)[1]
        return self.encode(src, src_mask), att

    def decode_and_attention(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        """Used at test-time only to retrieve attention matrix."""

        # Use first decoder layer
        decoder = self.transformer.decoder.layers[0]

        x = self.positional_encoding(self.tgt_tok_emb(tgt))
        x = decoder.norm1(x + decoder._sa_block(x, tgt_mask, None))
        att = decoder.multihead_attn(x, memory, memory, need_weights=True)[1]

        return self.transformer.decoder(x, memory, tgt_mask), att


# %% [markdown]
# ## Mask function

# %%
def create_mask(src: Tensor, tgt: Tensor):
    # Lengths of source and target sequences
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Attention mask for the source. As we have no reason to mask input
    # tokens, we use a mask full of False. You can use `torch.full`.
    # <answer>
    src_mask = torch.full((src_seq_len, src_seq_len), False)
    # </answer>

    # Attention mask for the target. To prevent a token from receiving
    # attention from future ones, we use a mask as defined in the lecture
    # (matrix `M`). You can use `torch.triu` and `torch.full` or directly
    # use the static function `generate_square_subsequent_mask` from the
    # `Transformer` class.
    # <answer>
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len)
    # </answer>

    # Boolean masks identifying tokens that have been padded with
    # `PAD_IDX`. Use `src` and `tgt` to create them. Don't forget to
    # ajust the size since both `src` and `tgt` are of size
    # `batch_size` * `seq_len` and the transformer object needs masks
    # of size `seq_len` * `batch_size`.
    # <answer>
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    # </answer>

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# %% [markdown]
# ## Training function

# %%
def train_epoch(model: nn.Module, dataset: Dataset, optimizer: Optimizer):
    # Training mode
    model.train()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    # <answer>
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # </answer>

    # Turn `dataset` into an iterable on mini-batches using `DataLoader`.
    # <answer>
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    # </answer>

    losses = 0
    for src, tgt in train_dataloader:
        # Select all but the last element of each sequence in `tgt`
        # <answer>
        tgt_input = tgt[:-1, :]
        # </answer>

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        scores = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        # Resetting gradients
        optimizer.zero_grad()

        # Select all but the first element of each sequence in `tgt`
        # <answer>
        tgt_out = tgt[1:, :]
        # </answer>

        # Permute dimensions before cross-entropy loss:
        #
        # - `logits` is `seq_length` * `batch_size` * `vocab_size` and should be
        #   `batch_size` * `vocab_size` * `seq_length`
        # - `tgt_out` is `seq_length` * `batch_size` and should be
        #   `batch_size` * `seq_length`
        # <answer>
        loss = loss_fn(scores.permute([1, 2, 0]), tgt_out.permute([1, 0]))
        # </answer>

        # Back-propagation through loss function
        loss.backward()

        # Gradient descent update
        optimizer.step()

        losses += loss.item()

    return losses / len(dataset)


# %% [markdown]
# ## Evaluation function

# %%
def evaluate(model: nn.Module, val_dataset: Dataset):
    model.eval()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    # <answer>
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # </answer>

    # Turn dataset into an iterable on batches
    # <answer>
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    # </answer>

    losses = 0
    for src, tgt in val_dataloader:
        # Select all but the last element of each sequence in `tgt`
        # <answer>
        tgt_input = tgt[:-1, :]
        # </answer>

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        # Select all but the first element of each sequence in `tgt`
        # <answer>
        tgt_out = tgt[1:, :]
        # </answer>

        # Permute dimensions for cross-entropy loss:
        #
        # - `logits` is `seq_length` * `batch_size` * `vocab_size` and should be
        #   `batch_size` * `vocab_size` * `seq_length`
        # - `tgt_out` is `seq_length` * `batch_size` and should be
        #   `batch_size` * `seq_length`
        # <answer>
        loss = loss_fn(logits.permute([1, 2, 0]), tgt_out.permute([1, 0]))
        # </answer>

        losses += loss.item()

    return losses / len(val_dataset)


# %% [markdown]
# ## Learning loop

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    FFN_HID_DIM,
)

optimizer = Adam(transformer.parameters(), lr=0.001)

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, train_set, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer, test_set)
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )


# %% [markdown]
# ## Helpers functions


# %%
def greedy_decode(model, src, src_mask, start_symbol_idx):
    """Autoregressive decoding of `src` starting with `start_symbol_idx`."""

    memory, att = model.encode_and_attention(src, src_mask)
    ys = torch.LongTensor([[start_symbol_idx]])
    maxlen = 100

    for i in range(maxlen):
        tgt_mask = Transformer.generate_square_subsequent_mask(ys.size(0))

        # Decode `ys`. `out` is of size `curr_len` * 1 * `vocab_size`
        out = model.decode(ys, memory, tgt_mask)

        # Select encoding of last token
        enc = out[-1, 0, :]

        # Get a set of scores on vocabulary
        dist = model.generator(enc)

        # Get index of maximum
        idx = torch.argmax(dist).item()

        # Add predicted index to `ys`
        ys = torch.cat((ys, torch.LongTensor([[idx]])))

        if idx == EOS_IDX:
            break
    return ys, att


def translate(model: torch.nn.Module, src_sentence: Iterable):
    """Translate sequence `src_sentence` with `model`."""

    model.eval()

    # Numericalize source
    src_tensor = torch.LongTensor(vocab_src(["<bos>"] + list(src_sentence) + ["<eos>"]))

    # Fake a minibatch of size one
    src = src_tensor.unsqueeze(-1)

    # No mask for source sequence
    seq_length = src.size(0)
    src_mask = torch.full((seq_length, seq_length), False)

    # Translate `src`
    tgt_tokens, att = greedy_decode(model, src, src_mask, BOS_IDX)

    tgt_tokens = tgt_tokens.flatten().numpy()
    att = att.detach().squeeze().numpy()
    return " ".join(vocab_tgt.lookup_tokens(list(tgt_tokens))), att


def plot_encoder_attention_matrix(model, src):
    """Plot heatmap of encoder's attention matrix."""

    model.eval()

    # Numericalize source
    src_delim = ["<bos>"] + list(src) + ["<eos>"]
    src_tensor = torch.LongTensor(vocab_src(src_delim))

    # Fake a minibatch of size one
    src = src_tensor.unsqueeze(-1)

    # No mask for source sequence
    seq_length = src.size(0)
    src_mask = torch.full((seq_length, seq_length), False)

    # Translate `src`
    memory, att = model.encode_and_attention(src, src_mask)

    ax = sns.heatmap(
        att.detach().squeeze().numpy(),
        xticklabels=src_delim,
        yticklabels=src_delim,
    )
    ax.set(xlabel='Key', ylabel='Query')

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
    )

def plot_decoder_attention_matrix(model, src, tgt):
    """Plot heatmap of decoder's cross-attention matrix."""

    model.eval()

    # Numericalize source and target
    src_delim = ["<bos>"] + list(src) + ["<eos>"]
    src_tensor = torch.LongTensor(vocab_src(src_delim))
    tgt_delim = ["<bos>"] + list(tgt) + ["<eos>"]
    tgt_tensor = torch.LongTensor(vocab_tgt(tgt_delim))

    # Fake a minibatch of size one
    src = src_tensor.unsqueeze(-1)
    tgt = tgt_tensor.unsqueeze(-1)

    # No mask for source sequence and triangular mask to target
    seq_length = src.size(0)
    src_mask = torch.full((seq_length, seq_length), False)
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt.size(0))

    # Encode `src`
    memory = model.encode(src, src_mask)

    # Retrieve cross-attention matrix
    _, att = model.decode_and_attention(tgt, memory, tgt_mask)

    ax = sns.heatmap(
        att.detach().squeeze().numpy(),
        xticklabels=src_delim,
        yticklabels=tgt_delim,
    )
    ax.set(xlabel='Key', ylabel='Query')

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
    )


src, tgt = test_set[0]
pred, att = translate(transformer, src)

plot_encoder_attention_matrix(transformer, src)
plt.show()

plot_decoder_attention_matrix(transformer, src, tgt)
plt.show()

# %% [markdown]
# # The transformer architecture from scratch

# %%
import math
from collections.abc import Iterable
from timeit import default_timer as timer
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

# %% [markdown]
# ## Toy dataset

# %%
def translate_deterministic(input_sequence):
    target_sequence = []
    for i, elt in enumerate(input_sequence):
        try:
            offset = int(elt)
        except ValueError:  # It is a letter
            target_sequence.append(elt)
        else:               # Special token, do the lookup
            if i + offset < 0 or i + offset > len(input_sequence) - 1:
                pass
            else:
                k = min(max(0, i + offset), len(input_sequence) - 1)
                target_sequence.append(input_sequence[k])

    return target_sequence


class GotoDataset(Dataset):
    def __init__(
        self,
        seed=None,
        n_sequences=100,
        min_length=4,
        max_length=20,
        n_letters=3,
        offsets=[4, 5, 6],
    ):
        super().__init__()
        full_vocab = "abcdefghijklmnopqrstuvwxyz"
        full_vocab = list(full_vocab.upper()) + list(full_vocab)
        assert(n_letters <= len(full_vocab))

        self.vocab = np.array(
            [s + str(d) for s in ["+", "-"] for d in offsets] + full_vocab[:n_letters]
        )
        self.n_tokens = len(self.vocab)
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed
        self.n_sequences = n_sequences

        # Dataset generation
        rs = np.random.RandomState(self.seed)
        seq_lengths = rs.randint(
            self.min_length, self.max_length, size=self.n_sequences
        )
        self.input_sequences = [
            list(self.vocab[rs.randint(self.n_tokens, size=seq_length)])
            for seq_length in seq_lengths
        ]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, i):
        input_sequence = self.input_sequences[i]
        target_sequence = translate_deterministic(input_sequence)
        return input_sequence, target_sequence

# %% [markdown]
# ## Vocabulary

# %%
dataset = GotoDataset()
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
vocab = build_vocab_from_iterator([dataset.vocab], specials=special_tokens)
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = vocab.lookup_indices(special_tokens)

# %% [markdown]
# ## Collate function

# %%
def collate_fn(batch: List):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:

        # Numericalize list of tokens using `vocab`.
        #
        # - Don't forget to add beginning of sequence and end of sequence tokens
        #   before numericalizing.
        #
        # - Use `torch.LongTensor` instead of `torch.Tensor` because the next
        #   step is an embedding that needs integers for its lookup table.
        # <answer>
        src_tensor = torch.LongTensor(vocab(["<bos>"] + src_sample + ["<eos>"]))
        tgt_tensor = torch.LongTensor(vocab(["<bos>"] + tgt_sample + ["<eos>"]))
        # </answer>

        # Append numericalized sequence to `src_batch` and `tgt_batch`
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    # Turn `src_batch` and `tgt_batch` that are lists of 1-dimensional
    # tensors of varying sizes into tensors with same size with
    # padding. Use `pad_sequence` with padding value to do so.
    #
    # Important notice: by default resulting tensors are of size
    # `max_seq_length` * `batch_size`; the mini-batch size is on the
    # *second dimension*.
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    # </answer>

    return src_batch, tgt_batch

# %% [markdown]
# ## Hyperparameters of transformer model

# %%
torch.manual_seed(0)

# Size of source and target vocabulary
VOCAB_SIZE = len(vocab)

# Number of sequences generated for the training set
N_SEQUENCES = 7000

# Number of epochs
NUM_EPOCHS = 20

# Size of embeddings
EMB_SIZE = 64

# Number of heads for the multihead attention
NHEAD = 1

# Size of hidden layer of FFN
FFN_HID_DIM = 128

# Size of mini-batches
BATCH_SIZE = 256

# Number of stacked encoder modules
NUM_ENCODER_LAYERS = 1

# Number of stacked decoder modules
NUM_DECODER_LAYERS = 1

# %% [markdown]
# ## Transformer encoder

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Define Tk/2pi for even k between 0 and `emb_size`. Use
        # `torch.arange`.
        # <answer>
        Tk_over_2pi = 10000 ** (torch.arange(0, emb_size, 2) / emb_size)
        # </answer>

        # Define `t = 0, 1,..., maxlen-1`. Use `torch.arange`.
        # <answer>
        t = torch.arange(maxlen)
        # </answer>

        # Outer product between `t` and `1/Tk_over_2pi` to have a
        # matrix of size `maxlen` * `emb_size // 2`. Use
        # `torch.outer`.
        # <answer>
        outer = torch.outer(t, 1 / Tk_over_2pi)
        # </answer>

        pos_embedding = torch.empty((maxlen, emb_size))

        # Fill `pos_embedding` with either sine or cosine of `outer`.
        # <answer>
        pos_embedding[:, 0::2] = torch.sin(outer)
        pos_embedding[:, 1::2] = torch.cos(outer)
        # </answer>

        # Add fake mini-batch dimension to be able to use broadcasting
        # in `forward` method.
        pos_embedding = pos_embedding.unsqueeze(1)

        self.dropout = nn.Dropout(dropout)

        # Save `pos_embedding` when serializing the model even if it is not a
        # set of parameters
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        # `token_embedding` is of size `seq_length` * `batch_size` *
        # `embedding_size`. Use broadcasting to add the positional embedding
        # that is of size `seq_length` * 1 * `embedding_size`.
        # <answer>
        seq_length = token_embedding.size(0)
        positional_encoding = token_embedding + self.pos_embedding[:seq_length, :]
        # </answer>

        return self.dropout(positional_encoding)

# %%
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            p=None,                  # Embedding size of input tokens
            d_ff=None,               # Size of hidden layer in MLP
    ):
        super().__init__()

        # Size of embedding. Here sizes of embedding, keys, queries
        # and values are the same.
        self.p = p
        d_q = d_v = d_k = p

        # Size of hidden layer in MLP
        self.d_ff = d_ff

        # Compute query, key and value from input
        self.enc_Q = nn.Linear(p, d_q)
        self.enc_K = nn.Linear(p, d_k)
        self.enc_V = nn.Linear(p, d_v)

        # Linear transform just before first residual mapping
        self.enc_W0 = nn.Linear(d_v, p)

        # Layer normalization after first residual mapping
        self.enc_ln1 = nn.LayerNorm(p)

        # Position-wise MLP
        self.enc_W1 = nn.Linear(p, d_ff)
        self.enc_W2 = nn.Linear(d_ff, p)

        # Final layer normalization of second residual mapping
        self.enc_ln2 = nn.LayerNorm(p)

    def forward(self, X):
        # Forward propagation in encoder. Input tensor `X` is of size
        # `seq_length` * `batch_size` * `p`.

        # Query, key and value of the encoder. Use `enc_Q`, `enc_K`
        # and `enc_V`.
        Q = ...
        K = ...
        V = ...

        # Score attention from `Q` and `K`. We need to compute `QK^T` but both
        # `Q` and `K` are not just simple matrices but batch of matrices. Both
        # `Q` and `K` are in fact of size `seq_length` * `batch_size` *
        # `emb_size`. Two ways to compute the batched matrix product:
        #
        # - permute dimensions using `torch.permute` so that `batch_size` is the
        #   first dimension and use `torch.bmm` that will perform the batch
        #   matrix product with respect to the first dimension,
        # - use `torch.einsum` to specify the product.
        S = ...

        # Compute attention from `S` and `V`. You can use `F.softmax` with `dim`
        # argument. Since the mini-batch dimension is now the first one for `S`
        # we can use `torch.bmm` with `S` (after softmax). That is not the case
        # for `V` so we need to transpose it first. Don't forget to transpose
        # again after the product to have a matrix `seq_length` * `batch_size` *
        # `emb_size` compatible with `X` for the residual mapping.
        A = ...
        T = ...

        # First residual mapping and layer normalization
        U = ...

        # FFN on each token
        Z = ...

        # Second residual mapping and layer normalization
        Xp = ...

        return Xp

# %% [markdown]
# ## Transformer decoder

# %%
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            p=None,                  # Embedding size of input tokens
            d_ff=None,               # Size of hidden layer in MLP
    ):

        super().__init__()

        # Size of embedding. Here, sizes of embedding, keys, queries
        # and values are the same.
        self.p = p
        self.d_q = self.d_v = self.d_k = p

        # Size of hidden layer in MLP
        self.d_ff = d_ff

        # Compute query, key and value from input
        self.dec_Q1 = nn.Linear(p, self.d_q)
        self.dec_K1 = nn.Linear(p, self.d_k)
        self.dec_V1 = nn.Linear(p, self.d_v)

        # Linear transform just before first residual mapping
        self.dec_W0 = nn.Linear(self.d_v, p)

        # Layer normalization after first residual mapping
        self.dec_ln1 = nn.LayerNorm(p)

        # Key-value cross-attention
        self.dec_Q2 = nn.Linear(p, self.d_k)
        self.dec_K2 = nn.Linear(p, self.d_k)
        self.dec_V2 = nn.Linear(p, self.d_v)

        # Linear transform just before first residual mapping
        self.dec_W1 = nn.Linear(self.d_v, p)

        # Layer normalization after second residual mapping
        self.dec_ln2 = nn.LayerNorm(p)

        # Position-wise MLP
        self.dec_W2 = nn.Linear(p, d_ff)
        self.dec_W3 = nn.Linear(d_ff, p)

        # Final layer normalization of second residual mapping
        self.dec_ln3 = nn.LayerNorm(p)

    def forward(self, Xp, Y):
        # Forward propagation in decoder. Input tensor `Xp` is of size
        # `seq_length_src` * `batch_size` * `p` and `Y` is of size
        # `seq_length_tgt` * `batch_size` * `p`.


        # Set number of tokens in target sequence `Y`. Needed to
        # compute the mask.
        m = Y.size(0)

        # Forward propagation of decoder. Use `dec_Q1`, `dec_K` and
        # `dec_V`.
        Q = ...
        K = ...
        V = ...

        # Compute square upper triangular mask matrix of size `m`. You
        # can use `torch.triu` and `torch.full` with `float("-inf")`.
        M = ...

        # Score attention from `Q` and `K`. You can use `torch.bmm`
        # and `transpose` but don't forget to add the mask `M`.
        S = ...

        # Attention
        A = ...
        T1 = ...

        # First residual mapping and layer normalization
        U1 = ...

        # Key-value cross-attention using keys and values from the
        # encoder.
        Q = ...
        K = ...
        V = ...

        # Score attention from `Q` and `K`. You can either use
        # `torch.bmm` together with `torch.permute` or `torch.einsum`.
        # S = torch.bmm(Q.permute([1, 0, 2]), K.permute([1, 2, 0])) / math.sqrt(self.p)
        S = ...

        # Attention
        A = ...
        T2 = ...

        # Second residual mapping and layer normalization
        U2 = ...

        # FFN on each token
        Z = ...

        # Third residual mapping and layer normalization
        U3 = ...

        return U3


# %% [markdown]
# ## Transformer model

# %%
class Transformer(nn.Module):
    def __init__(self, p=None, d_ff=None, vocab_size=None):
        super().__init__()

        # Declare an embedding, a positional encoder and a transformer
        # encoder.
        self.enc_embedding = nn.Embedding(vocab_size, p)
        self.enc_positional_encoding = PositionalEncoding(p)
        self.encoder = TransformerEncoder(p=p, d_ff=d_ff)

        # Declare an embedding, a positional encoder and a transformer
        # decoder.
        self.dec_embedding = nn.Embedding(vocab_size, p)
        self.dec_positional_encoding = PositionalEncoding(p)
        self.decoder = TransformerDecoder(p=p, d_ff=d_ff)

        self.generator = nn.Linear(p, vocab_size)

    def encode(self, X):
        # Use `self.enc_embedding`, `self.enc_positional_encoding` and
        # `self.encoder` to compute `Xp`
        X_emb = ...
        X_emb_pos = ...
        Xp = ...
        return Xp

    def decode(self, Xp, Y):
        # Use `self.dec_embedding`, `self.dec_positional_encoding` and
        # `self.decoder` to compute `outs`
        Y_emb = ...
        Y_emb_pos = ...
        outs = ...
        return outs

    def forward(self, X, Y):
        Xp = self.encode(X)
        outs = self.decode(Xp, Y)
        return self.generator(outs)


def train_epoch(model: nn.Module, dataset: Dataset, optimizer: Optimizer):
    # Training mode
    model.train()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    loss_fn = ...

    # Turn `dataset` into an iterable on mini-batches using `DataLoader`.
    train_dataloader = ...

    losses = 0
    for X, Y in train_dataloader:
        # Select all but last element in sequences
        Y_input = ...

        # Resetting gradients
        optimizer.zero_grad()

        # Compute output of transformer from `X` and `Y_input`.
        scores = ...

        # Back-propagation through loss function
        # Select all but first element in sequences
        Y_output = ...

        # Compute the cross-entropy loss between `scores` and
        # `Y_output`. `scores` is `seq_length` * `batch_size` *
        # `vocab_size` and contains scores and `Y_output` is
        # `seq_length` * `batch_size` and contains integers. Two ways
        # to compute the loss:
        #
        # - reshape both tensors to have `batch_size` * `probs` for `scores` and
        #   `batch_size` for `Y_output`
        # - permute dimensions to have `batch_size` * `vocab_size` *
        #   `seq_length` for `scores` and `batch_size` * `seq_length` for
        #   `Y_output`
        loss = ...

        # Gradient descent update
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(dataset)


# %% [markdown]
# ## Eval function

# %%
def evaluate(model: nn.Module, val_dataset: Dataset):
    model.eval()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    loss_fn = ...

    # Turn `val_dataset` into an iterable on mini-batches using `DataLoader`.
    val_dataloader = ...

    losses = 0
    for X, Y in val_dataloader:
        # Select all but last element in sequences
        Y_input = ...

        # Compute output of transformer from `X` and `Y_input`.
        scores = ...

        # Select all but first element in sequences
        Y_output = ...

        # Compute loss
        loss = ...

        losses += loss.item()

    return losses / len(val_dataset)


# %% [markdown]
# ## Learning loop

# %%
transformer = Transformer(
    p=EMB_SIZE,
    d_ff=FFN_HID_DIM,
    vocab_size=VOCAB_SIZE
)

optimizer = Adam(transformer.parameters())

train_set = GotoDataset(n_sequences=N_SEQUENCES)
test_set = GotoDataset(n_sequences=N_SEQUENCES)

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, train_set, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer, test_set)
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )


# %% [markdown]
# ## Helpers functions

# %%
def greedy_decode(model, src, start_symbol_idx):
    """Autoregressive decoding of `src` starting with `start_symbol_idx`."""

    memory = model.encode(src)
    ys = torch.LongTensor([[start_symbol_idx]])
    maxlen = 100

    for i in range(maxlen):
        m = ys.size(0)
        tgt_mask = torch.triu(torch.full((m, m), float("-inf")), diagonal=1)

        # Decode `ys`. `out` is of size `curr_len` * 1 * `vocab_size`
        out = model.decode(memory, ys)

        # Select encoding of last token
        enc = out[-1, 0, :]

        # Get a set of scores on vocabulary
        dist = model.generator(enc)

        # Get index of maximum
        idx = torch.argmax(dist).item()

        # Add predicted index to `ys`
        ys = torch.cat((ys, torch.LongTensor([[idx]])))

        if idx == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: Iterable):
    """Translate sequence `src_sentence` with `model`."""

    model.eval()

    # Numericalize source
    src_tensor = torch.LongTensor(vocab(["<bos>"] + list(src_sentence) + ["<eos>"]))

    # Fake a minibatch of size one
    src = src_tensor.unsqueeze(1)

    # Translate `src`
    tgt_tokens = greedy_decode(model, src, BOS_IDX)

    tgt_tokens = tgt_tokens.flatten().numpy()
    return " ".join(vocab.lookup_tokens(list(tgt_tokens)[1:-1]))


input, output = dataset[2]

print("Input:", " ".join(input))
print("Output:", " ".join(output))
print("Pred:", translate(transformer, input))
