# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os, cv2, urllib, urllib.request, zipfile


"""OBJECTS
"""
class Layer_Dense:

    """
    Layer of neurons, where each neuron posesses an associated weight and bias value. Weights are between the neuron itself,
    and either another neuron from a different layer, or an input value. Bias values are unique to each neuron. Both weight values
    and biases will be updated by the optimizer, based on the models loss and dvalues for each neuron.
    Dense layers perform two functions:
    1) Forward pass, where the general calculation is as follows: (input value * weight) + bias = output.
    The output value will then be passed to the activation function, which determines whether the neuron fires or not.
    2) Backward pass, where....
    """

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0., weight_regularizer_l2=0., bias_regularizer_l1=0., bias_regularizer_l2=0.):

        # Initializing random weights
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initial bias set as 0
        self.biases = np.zeros((1, n_neurons))

        """
        Regularization error is a product of 'overtraining' on sample data. Reducing regularization error allows a model
        to be more accurate on testing data ie. samples that the model hasn't seen.
        This works by penalizing a model for having large magnitude weights and biases. When these magnitude values are too
        high, that is a sign that the model is attempting to memorize training data (and therefore will perform poorly
        on testing data).
        l1 is the sum of the weight values (linear penalty).
        l2 is the product of the weight values (non linear penalty).
        The regularization penalty is added to the total loss (error of the model). 
        """
        # Initial variable values for regularization set as 0
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):

        # Remember input
        self.inputs = inputs

        # Output calculation. Dot product is used to sum the product of the input and weights
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        """
        The backward pass uses partial derivitives to determine the extent to which this neuron contributed
        to the model's loss. That 'gradient' can then be reapplied to the weight, in order to reduce loss.
        """
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        """
        Below formulas derived from textbook derivative calculations. Amount to the regularization portion.
        """
        # Gradient on regularization
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:

    """
    Dropout layer will disable certain neurons in an attempt to prevent a network from becoming overdependent on a
    certain set of neurons i.e. the network must become more generalized.
    This is another type of regularization.
    The rate at which neurons are switched off is the parameter 'rate'. Disabling neurons is done randomly using the
    Bernoulli distribution. The dropout will apply to the following layer.
    """

    def __init__(self, rate):

        # Inverted store rate
        self.rate = 1 - rate

    def forward(self, inputs, training):

        # Save input values
        self.inputs = inputs

        # If not in training mode, then return values as they are
        if not training:
            self.output = inputs.copy()

            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):

        # Same binary mask gradient on dvalues
        self.dinputs = dvalues * self.binary_mask

class Layer_Input:

    # Forward pass only
    def forward(self, inputs, training):

        self.output = inputs


class Activation_ReLU:

    """
    Activation functions determine whether the neuron will fire on not.
    The output from the neuron must be above a certain amount in order for it to fire.
    Different activation functions control what the output must be in order for the neuron to fire.
    ReLU (Rectified Linear) uses a 'floor' of 0. Anything below 0 will be 0.
    Anything above will be the original value. i.e. no negative values
    ReLU is one of the most powerful and widely used activation functions.
    """

    def forward(self, inputs, training):

        # Remember input values for use in backward pass
        self.inputs = inputs

        # Calculate output
        self.output = np.maximum(0, inputs) # Main code performing forward pass logic i.e. values cannot be below 0.

    def backward(self, dvalues):

        # Create a copy of original variable in order to safely modify it
        self.dinputs = dvalues.copy()
        # Derivation is simply to set values as 0 where inputs were 0. This is why we needed to keep a copy of the inputs
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):

        return outputs

class Activation_Softmax:

    """
    Activation Softmax is used for classification i.e. however many outcomes are expected in the final prediction.
    It first takes the exponential of each output value and then normalizes them by diving by the sum of expenentials.
    The result is a probability distribution which will be used as a confidence score for each class.
    """

    def forward(self, inputs, training):

        # Remember input values
        self.inputs = inputs

        # Calculate unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        # Create empty array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate through outputs and gradients
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

    def predictions(self, outputs):

        return np.argmax(outputs, axis=1)

class Loss:

    """
    Generic loss class to perform functions relevent to all other loss calculation classes.
    """

    # Regularization loss
    def regularization_loss(self):

        # Starting at 0
        regularization_loss = 0

        # Iterate through layers
        for layer in self.trainable_layers:

            # l1 weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            # l2 weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * (np.sum(layer.weights * layer.weights))
            # l1 biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            # l2 biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * (np.sum(layer.biases * layer.biases))

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):

        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # If no regularization loss required, then return only data loss
        if not include_regularization:

            return data_loss

        return data_loss, self.regularization_loss()

class Loss_CategoricalCrossEntropy(Loss):

    """
    Loss calculation using natural logrithm as its basis. Convenient for 'one-hot' predictions.
    """

    def forward(self, y_pred, y_true):

        # Determine number of samples
        samples = len(y_pred)
        # Clip to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Final loss calculation
        negative_log_likelihoods = np.log(correct_confidences)

        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        # If labels are sparce, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy:

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Accuracy:

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        # Calculate
        accuracy = np.mean(comparisons)

        return accuracy

class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):

        self.binary = binary

    def compare(self, predictions, y):

        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y

class Optimizer_Adam:

    """
    Optimizers adjust the neuron weights and biases, based on the backward pass gradient, in order to decrease loss.
    Learning rate determines how impactful each iteration is in changing the parameters.
    Decay reduces the learning rate over time, allowing the model to fine tune and make smaller changes when it is closer
    to optimization.
    Epsilon, Beta 1 and 2 all have to do with momentum i.e. being able to blast through local minimums.
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentums with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Corrected momentums
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Update and normalization
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


    def post_update_params(self):

        self.iterations += 1

class Model:

    """
    General usage model class.
    """

    def __init__(self):

        # Blank list to contain layers added to NN
        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):

        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):

        self.input_layer = Layer_Input()

        # Count of layers
        layer_count = len(self.layers)
        # Blank list of trainable
        self.trainable_layers = []

        for i in range(layer_count):

            # If first layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # Last layer
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Update loss with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        for epoch in range(1, epochs + 1):

            # Forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Backward pass
            self.backward(output, y)

            # Optimize
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:

                self.optimizer.update_params(layer)

            self.optimizer.post_update_params()

            # Summary print
            if not epoch % print_every:
                print(f"Epoch: {epoch}, " +
                      f"Acc: {accuracy: .3f}, " +
                      f"Loss: {loss: .3f}, " +
                      f"Data Loss: {data_loss: .3f}, " +
                      f"Reg Loss: {regularization_loss: .3f}, " +
                      f"Learning Rate: {self.optimizer.current_learning_rate: .3f}, ")

        # Final validation on testing data
        if validation_data is not None:
            X_val, y_val = validation_data

            # Forward pass
            output = self.forward(X_val, training=False)
            # Loss, Accuracy and Preds
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Final print
            print("-" * 50)
            print(f"Validation, " +
                  f"Acc: {accuracy: .5f}, " +
                  f"Loss: {loss: .5f}")

    def forward(self, X, training):

        self.input_layer.forward(X, training)

        for layer in self.layers:

            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):

                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):

            layer.backward(layer.next.dinputs)

# Sample data creation
URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}...")
    urllib.request.urlretrieve(URL, FILE)

    print("Unzipping images...")
    with zipfile.ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
        print("Done!")

"""RUN
"""
if __name__ == "__main__":

    # Read in sample data
    labels = os.listdir("fashion_mnist_images/train")

    X = []
    y = []
    # Loop through label folders
    for label in labels:

        print(f"Collecting files with label: {label} for training data")
        # Loop through each file and append image using cv2
        for file in os.listdir(os.path.join("fashion_mnist_images", "train", label)):

            image = cv2.imread(os.path.join("fashion_mnist_images", "train", label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    X_test = []
    y_test = []
    # Loop through label folders
    for label in labels:

        print(f"Collecting files with label: {label} for testing data")
        # Loop through each file and append image using cv2
        for file in os.listdir(os.path.join("fashion_mnist_images", "test", label)):

            image = cv2.imread(os.path.join("fashion_mnist_images", "test", label, file), cv2.IMREAD_UNCHANGED)

            X_test.append(image)
            y_test.append(label)


    # Initialize Model
    model = Model()
    model.add(Layer_Dense(28, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(rate=0.1))
    model.add(Layer_Dense(512, 10))
    model.add(Activation_Softmax())

    # Set
    model.set(loss=Loss_CategoricalCrossEntropy(),
              optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
              accuracy=Accuracy_Categorical())

    # Finalize and train
    model.finalize()
    model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)