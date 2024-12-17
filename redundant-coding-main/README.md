
# `ResNetVAE/modules.py`
This code defines a neural network model called ResNet_VAE that is capable of encoding and decoding image data using a variational autoencoder architecture. The model uses a pre-trained ResNet18 model for encoding, and includes several fully connected layers and convolutional layers for decoding. The model is capable of generating a reconstructed image from an input image, as well as outputting a latent representation of the input image that can be used for further downstream analysis.

# `ResNetVAE/ResNetVAE_cifar10.py`
The ResNetVAE_cifar10.py script contains the code to train a variant of the Variational AutoEncoder using the ResNet architecture as an encoder to encode an image into its latent representation. The script uses the CIFAR10 dataset to train the model and record the loss metrics throughout the training process. The code handles saving and loading of model and optimizer states, and uses weights and biases for logging and visualization. Finally, the trained models are stored in local disk space as well. The script has an adjustable dropout probability and can be trained for an adjustable number of epochs.

# `ResNetVAE/make_plots.py`
This code is used to analyze the properties of a ResNet-VAE neural network model trained on the CIFAR-10 dataset. The code first loads the model and uses hooks to extract activations at specified layers. The activations are then used to perform PCA analysis and the variance explained by each component is plotted. The code then generates activations for a set of dot images and project them onto a low-dimensional space obtained through a random projection matrix. The energy of the projection and the correlation between inputs and the projection are computed and plotted. Finally, the code computes a decay coefficient for each layer of the model and plots it against the energy of the projection and the correlation between inputs and the projection. The results are saved to disk as figures.

# `ResNetVAE/helpers.py`
This code contains several utility functions for computer vision and machine learning tasks. It includes a function to generate a sequence of dot images moving in a circular path, and preprocesses the images using the default transform for CIFAR images. Another function computes the total energy of a set of points in n-dimensional space by calculating the squared magnitude of the differences between successive points. Additionally, there is a function that returns a dictionary of random projection matrices with specified dimensions for different layers in a neural network. These functions can be used to generate image data, calculate energy, and create random projection matrices for machine learning tasks.

# `automated-interpretability/neuron-explainer/generate_trajectory.py`
This code generates random trajectories for a pre-trained language model. It captures activations of certain layers during a forward pass and computes the total energy of a set of points in n-dimensional space. It then finds the minimum trajectory and project it to a 1D plot. The process is repeated multiple times to visualize different trajectories in each iteration. The final result is a set of PDF files and NPY files showing the 1D projections of different minimum trajectories. The generated plots can be used to gain insights into how the language model processes various sequences of text inputs.

# `automated-interpretability/neuron-explainer/generate_explanations.py`
This script generates explanations for the neuron activation patterns of a pre-trained GPT-2 XL language model using a calibrated simulator. The explanations are generated for the top 10 activation patterns of each layer in the neural network based on their mean maximum activation value compared to all activation patterns. The TokenActivationPairExplainer is used to generate explanations for the neuron activations, followed by simulating and scoring the explanation using an UncalibratedNeuronSimulator. The generated explanations and scores are saved for later use.

# `MINE.py`
This script is a simply tensorflow implementation of the originaly MINE paper. Using a neural network to maximise a certain form of the mutual information we can bound the continious analogue of the channel capacity of a given channel.

# `Estimate-MI-of-NN-Layers.py`
This script trains a very deep neural network on a simple task (preserving information) in a regime where error correction is needed to preserve any information. The script contains the functionality to train a model for this task, to deploy MINE to estimate the MI of each layer, and to plot these results to recreate plots shown in the paper.
