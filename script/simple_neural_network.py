import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import  tqdm
import joblib
import cv2

plt.rcParams['figure.figsize'] = [17, 10]


def load_model(x):
    """
    Charge un model sauvegarder au format .sav (joblib)
    """
    return joblib.load(x)


def generate_batch(X, Y, batch_size):
    """
    Generate a batch for X and Y of corresponding indexes
    """
    batch = np.random.choice(X.shape[0], batch_size)
    return X[batch], Y[batch]


def ponderation(X, W, b):
    """
    1 ponderation
    return : output layer
    """
    return np.dot(X, W)+b


def relu(x):
    """
    Relu activation function
    return : relu output (max(0,x))
    """
    return np.maximum(0, x)


def softmax(scores, eps):
    """
    Compute class probabilities by softmax
    return : sofwtmax output (probabilities)
    """
    exp = np.exp(scores)
    probs = exp/(np.sum(exp, axis=1, keepdims=True)+eps)
    return probs


def cross_entropy(probs, y, num_examples, eps):
    """
    Compute the loss: average cross-entropy loss and regularization
    return : cross_entropy loss
    """
    correct_logprobs = -np.log(probs[range(num_examples), y])+eps
    data_loss = np.sum(correct_logprobs)/num_examples

    return data_loss


def get_accuracy(probs, y):
    """
    Compute accuracy after ponderation
    Do a comparaison between the predicted value and the true value
    return : accuracy (from 0 to 1)
    """
    predict_class = np.argmax(probs, axis=1)
    return np.mean(predict_class == y)


def grad(probs, y, num_examples):
    """
    Compute the gradient on probs
    return : dscores
    """
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    return dscores


def backpropagation(hidden_layer, dscores, W2, W, X):
    """
    backpropate the gradient to the parameters
    return : dW,db,dW2,db2
    """
    # first backprop into parameters W2 and b2
    dW2 = ponderation(hidden_layer.T, dscores, b=0)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = ponderation(dscores, W2.T, b=0)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = ponderation(X.T, dhidden, b=0)
    db = np.sum(dhidden, axis=0, keepdims=True)

    return dW, db, dW2, db2


def updates(W, b, W2, b2, dW, db, dW2, db2, learning_rate):
    """
    updates the weights
    return : W,b,W2,b2
    """
    W += -learning_rate * dW
    b += -learning_rate * db
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2

    return W, b, W2, b2


def invert(img):
    """
    Invert an 8 bit image from Microsoft Paint ()
    """
    output = -img+255
    return output


class NN():
    def __init__(self, X_train, X_test, Y_train, Y_test, neurons, epochs, learning_rate, seed=0, batch_size=1024,verbose=True):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.input_dim = 28*28
        self.ouput_dim = 10

        """
        Hyperparametres
        """
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.batch_size = batch_size

        """ Xavier Initialization
        W=np.random.randn(self.input_dim, self.neurons)/np.sqrt(input_dim)
        """

        """
        He Initialization (taking Relu in count)
        W=np.random.randn(self.input_dim, self.neurons)/np.sqrt(input_dim/2)
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.W = 0.01 * np.random.randn(self.input_dim, self.neurons)
        self.b = np.zeros((1, self.neurons))
        self.W2 = 0.01 * np.random.randn(self.neurons, self.ouput_dim)
        self.b2 = np.zeros((1, self.ouput_dim))

        """
        Logging data
        """
        self.train_loss = np.zeros(self.epochs)
        self.train_accuracy = np.zeros(self.epochs)
        self.test_loss = np.zeros(self.epochs)
        self.test_accuracy = np.zeros(self.epochs)

        self.num_examples = self.X_train.shape[0]
        self.num_examples_test = self.X_test.shape[0]
        self.verbose=verbose

    def predict(self, X):
        """
        Prediction
        return : predicted_class,probability
        """
        hidden_layer = relu(ponderation(X, self.W, self.b))
        scores = softmax(ponderation(hidden_layer, self.W2, self.b2), self.eps)
        predicted_class = np.argmax(scores, axis=1)
        probability = np.max(scores)
        return predicted_class, probability

    def img_pred(self, X):
        """
        Predict the class and probability of an image X
        return : image,predicted_class,probability
        """
        img = invert(cv2.imread(X, 0))
        plt.imshow(img, cmap="gray")
        img = img.reshape(28*28)
        return self.predict(img)

    def loss_curve(self, return_fig=False, diff=False):
        """
        Plot the loss curve over the epochs.
        Can show the difference between train and test
        """
        fig1, ax1 = plt.subplots()
        ax1.plot(range(self.epochs),
                 self.train_loss,
                 label="train loss")

        ax1.plot(range(self.epochs),
                 self.test_loss,
                 label="test loss",
                 color='red')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        h1, l1 = ax1.get_legend_handles_labels()
        plt.title("Loss Curve")

        if diff:
            ax2 = ax1.twinx()
            ax2.plot(range(self.epochs),
                     np.abs(self.train_loss-self.test_loss),
                     label="Loss diff", color='green')

            ax2.set_ylabel('Loss diff')
            ax1.legend(h1+h2, l1+l2, loc=0)
            h2, l2 = ax2.get_legend_handles_labels()
        else:
            ax1.legend(h1, l1, loc=0)

        if return_fig:
            return fig1
        else:
            return

    def accuracy_curve(self, return_fig=False, diff=False):
        """
        Plot the accuracy curve over the epochs.
        Can show the difference between train and test
        """
        fig1, ax1 = plt.subplots()
        ax1.plot(range(self.epochs),
                 self.train_accuracy,
                 label="Train accuracy")

        ax1.plot(range(self.epochs),
                 self.test_accuracy,
                 label="Test accuracy",
                 color='red')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        plt.title("Accuracy Curve")
        h1, l1 = ax1.get_legend_handles_labels()

        if diff:
            ax2 = ax1.twinx()
            ax2.plot(range(self.epochs),
                     np.abs(self.train_accuracy - self.test_accuracy),
                     label="Accuracy diff", color='green')

            ax2.set_ylabel('Accuracy diff')
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc=0)
        else:
            ax1.legend(h1, l1, loc=0)

        if return_fig:
            return fig1
        else:
            return

    def save(self, name):
        """
        Save the current model as a .sav file (joblib)
        """
        return joblib.dump(self, name)

    def score(self, X, Y):
        """
        Get the accuracy of the model on a new dataset
        return : accuracy (from 0 to 1)
        """
        predict_class, _ = self.predict(X)
        accuracy = np.mean(predict_class == Y)
        return accuracy

    def fit(self):
        """
        Training process
        """
        for i in tqdm(range(self.epochs)):

            X_batch, Y_batch = generate_batch(self.X_train,
                                              self.Y_train,
                                              self.batch_size)

            X_test_batch, Y_test_batch = generate_batch(self.X_test,
                                                        self.Y_test,
                                                        self.batch_size)

            # 1st layer + relu
            hidden_layer = relu(ponderation(X_batch, self.W, self.b))

            # 2nd layer
            scores = ponderation(hidden_layer, self.W2, self.b2)

            # softmax
            probs = softmax(scores, self.eps)

            # crossentropy loss
            loss = cross_entropy(probs, Y_batch, self.batch_size, self.eps)

            # logging loss
            self.train_loss[i] = loss

            # logging accuracy
            self.train_accuracy[i] = get_accuracy(probs, Y_batch)

            # test set logging
            hidden_layer_test = relu(ponderation(X_test_batch, self.W, self.b))
            scores_test = ponderation(hidden_layer_test, self.W2, self.b2)
            probs_test = softmax(scores_test, self.eps)
            loss_test = cross_entropy(probs_test,
                                      Y_test_batch,
                                      self.batch_size,
                                      self.eps)

            self.test_accuracy[i] = get_accuracy(probs_test, Y_test_batch)
            self.test_loss[i] = loss_test

            # gradient
            dscores = grad(probs, Y_batch, self.batch_size)

            # backpropagation
            dW, db, dW2, db2 = backpropagation(hidden_layer,
                                               dscores,
                                               self.W2,
                                               self.W,
                                               X_batch)

            # updates
            self.W, self.b, self.W2, self.b2 = updates(self.W,
                                                       self.b,
                                                       self.W2,
                                                       self.b2,
                                                       dW,
                                                       db,
                                                       dW2,
                                                       db2,
                                                       self.learning_rate)
            if self.verbose:
                if i % (self.epochs/20) == 0:
                    print(
                        f"Epoch {i: <4}  train_loss : {round(self.train_loss[i],4): <9}  test_loss : {round(self.test_loss[i],4): <9}  train_accuracy : {round(self.train_accuracy[i],4): <9}  test_accuracy : {round(self.test_accuracy[i],4): <9}")