r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**




"""

part1_q2 = r"""
**Your answer:**




"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 1, 0.05, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_vanilla = 0.03
    lr_momentum = 0.003
    lr_rmsprop = 0.00021544346900318823  #There is a truly marvelous way to derive this value but this comment is too small to contain it.
    wstd = 0.2
    reg = 0.001584893192461114
    '''lr_vanilla = 0.02
    lr_momentum = 0.002
    lr_rmsprop = 0.0002
    wstd = 0.2
    reg = 0.002'''
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.002
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**




"""

part2_q2 = r"""
**Your answer:**




"""

part2_q3 = r"""
**Your answer:**




"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 8
    activation = 'relu'
    out_activation = 'none'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.003
    weight_decay = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**




"""

part3_q2 = r"""
**Your answer:**




"""

part3_q3 = r"""
**Your answer:**




"""


part3_q4 = r"""
**Your answer:**




"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.01, 0.001, 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

**1-**
**Vanilla resnet block**
after each convolution we have (channels-out * ((channels_in * width * height) + 1) parameters.
$K \cdot (C_in \cdot F^2 + 1)$

Thus
First convolution:
Parameters = $256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
Second convolution:
Parameters = $256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
Total parameters = 590,080 $\cdot$ 2 = 1,180,160
   
*BottleNeck block*
First convolution:
parameters = $64 \cdot ((256 \cdot 1 \cdot 1) + 1) = 16,448 $
   
Second convolution:
parameters = $64 \cdot ((64 \cdot 3 \cdot 3) + 1) = 36,928 $
   
Third convolution:
parameters = $256 \cdot ((64 \cdot 1 \cdot 1) + 1) = 16,640 $
   
Total parameters = 16,448 + 36,928 + 16,640 = 70,016 
   
We can see that in bottleneck there are fewer parameters

**2-**
TODO--------------------------------------------------------------------------------------
should we include bias? ($C_{in}\times (k^2 + 1)$)
------------------------------------------------------------------------------------------
For each filter, for each stride we have $C_{in}\times k^2$ multiplication operations $C_{in}\times (k^2 -1)$ addition operations and 1 bias adding operation.
Therefore we have $2\times C_{in}\times k^2$ operations per filter $\cdot$ stride.
We have $H\times W$ strides, so we have  $2\times C_{in}\times k^2\times C_{out}\times H\times W$  operations per layer. 
For each layer we denote the number of params $P_l = C_{in}\times k^2\times C_{out}$  
Thus, the number of floating point operations for each layer is  $2\times P_l\times H\times W$

shortcut connections require $C_{out}\times H \times W$ Addition operations


To conclude the final equation of floating point operations would be:
floating point operations = $C_{out}\times H \times W +\sum_{l\in layers} 2\times P_l\times H \times W$
Using dimension preserving padding, we get constant HxW thus we can simplify:
floating point operations = $C_{out}\times H \times W +2\times H \times W\sum_{l\in layers}P_l$
floating point operations = $H \times W(C_{out} + 2\times\sum_{l\in layers}P_l)$
denoting $P_b$ as number of parameters in block we get
floating point operations = $H \times W(C_{out} + 2\times P_b)$

In vanilla block we have floating point operations = $H \times W(256+2\times 1,180,160) = 2,360,576\times H\times W$
In the bottleneck block we have floating point operations = $H\times W (256+2\times 70,016) =  H\times W \times 140,288 $

We can see that bottleneck requires much less floating point operations to compute an output

**3-** 
Spatial - 
    regular block - we use two convolution layers of 3x3 thus we get respective field of 5x5.
    bottleneck block -  we use two convolution layers of 1x1 and one convolution layer of 3x3 thus we get
        respective field of 3x3.
   
We see that vanilla block combines the input better in terms of spatial.
   
   
Across feature map-   
   In bottleneck block not all inputs has the same influence across feature map, that because we project the first layer to a smaller dimension
   In vanilla block we don't project the input (therefore we have the same influence)
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1. We got the best accuracy by using L=4.
We see that when using L=2 and L=4 the results are almost the same.
As we learned in class - deeper network (with more layers) are more complex and can fit better to general data (up to a certen level, where we get overfit)

2. The network was untrainable for depths 8 and 16 - the largest depths. 
We think that happened because of vanishing gradients - the info flowing in the network passed a lot of loss layers, and it makes the gradient zero.
"""

part5_q2 = r"""
**Your answer:**
When looking at both 1.1 and 1.2 experiments - we can see that both networks aren't trainable with L=8 due to vanishing gradients.
We also see that with L=4 we get better test accuracy for every K tested. Which suitable for what we got in experiment 1.1

We see that for L=4 for more filters per layers we got better test accuracy whereas for L=2 the fewer filters per layer (on number tested) get better test accuracy.




"""

part5_q3 = r"""
**Your answer:**
We can see that for L=4 - we got vanishing gradients and the network was untrainable.

As we saw in the previous experiments for a fixed k we get better test accuracy for higher amount of layers,
but we can also see that after adding too much architecture complexity, the networks can be unstable and become untrainable.
"""

part5_q4 = r"""
We can see that the model was not trainable with K=32, our guess is that it might be because of the momentum and that
we jump between local minimas.
We can see that for L=2 and k=[64, 128, 256] the model was trainable and for L=4 and L=8, it wasnt. we balieve it for
the same reasons as the previous experiments.




"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

For the first picture:
1. The model didn't recognize the dolphins present in the image. Instead of it, it detects a bird and a person.
2. The reason for this is that "dolphin" is not a possible class for an object in YoloV3 model. It probably detects a 
bird because of the sky in the background and the black color of the dolphin. The method I would suggest to recognize 
dolphin in the image would be to add some dolphin images to the dataset, a new dolphin class and retrain the model with
it. I would also suggest having some shadow images of dolphins in the new dataset to be able to recognize them.

For the second picture:
1. The model detected the objects almost right. In right side, it detects two dogs and draw the relevant bounding boxes
around them. In left side, it detects a cat, which is present in the image, but draw the bounding box mainly on the
third dog present left and not around the cat. So it missed the right bounding box around the cat and the detection of
the third dog.
2. I would say that the reason why the models didn't detects well the object on right is because the image is cluttered,
there is a lot of objects and it's difficult for the model to detect each one of them.
(TODO: check this with team)
"""


part6_q2 = r"""
**Your answer:**




"""


part6_q3 = r"""
**Your answer:**

For the first picture - Illumination conditions:
The model detects the bottle present in the image as a cup.
Although 'bottle' is in the classes names, the model detects it as a cup.
It's because of the lighting condition, we have a low 
light (I needed to shut down the light in the room :)) and maybe also because of the clutter in the background
(multiple objects to detect in a small area).
Yet,it identifies the tv monitor, keyboard and mouse correctly. 

For the second picture - Occlusion:
The model didn't detect the 'hair drier' in the picture even though it is in the classes names.
We believe that the reason is because it's cut in the image, and partially occlude, and thus missing important features.

For the third picture - Cluttering:
Crowded or Cluttered Scenario: Too many objects in the image make it extremely crowded.
We can see that the model can detect apples but it didnt detect all the apples.
It's not obvious there should be apple in the air. It detects some apple clustered together on the table and in the
floor box but there is a lot of missing. We think the reason for this is that there is too much object to detect in this
image. Also, it was able to detect other objects. 



"""

part6_bonus = r"""
**Your answer:**




"""