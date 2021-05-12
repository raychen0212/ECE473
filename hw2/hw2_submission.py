import collections
import math
import numpy as np
import re
import string


class Logistic_Regression():
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, x, y, batch_size=64, iteration=2000, learning_rate=1e-2):
        """
        Train this Logistic Regression classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """
        dim = x.shape[1]
        num_train = x.shape[0]

        # initialize W
        if self.W == None:
            self.W = 0.001 * np.random.randn(dim, 1)
            self.b = 0

        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and update W, b

            correct = 0
            error = 0
            x_batch = (x_batch-np.mean(x_batch))/np.std(x_batch)
            # print(y_pred_loss)
            y_pred = self.predict(x_batch)
            # print(y_pred_loss)
            ###########
            y_calc = np.dot(x_batch, self.W) + self.b
            y_loss = self.sigmoid(y_calc)
            loss, grad_dict = self.loss(x_batch, y_loss, y_batch)
            for i in range(len(grad_dict['dW'])):
                self.W[i] -= learning_rate * grad_dict['dW'][i]
            for b in range(len(grad_dict['db'])):
                self.b -= learning_rate * grad_dict['db'][b]
            for k in range(len(y_batch)):
                if y_batch[k] == y_pred[k]:
                    correct += 1
                else:
                    error += 1
            #acc = np.mean(y_pred==y_batch)
            acc = correct / (correct + error)
            # END_YOUR_CODE
            ############################################################
            ############################################################

            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate predicted y
        x = (x - np.mean(x)) / np.std(x)
        y = np.dot(x, self.W) + self.b
        y_pred = Logistic_Regression.sigmoid(self, y)
        for i in range(len(y_pred)):
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            elif y_pred[i] < 0.5:
                y_pred[i] = 0
        # pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_pred

    def loss(self, x_batch, y_pred, y_batch):
        """
        Compute the loss function and its derivative. 
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient dictionary with two keys : 'dW' and 'db'
        """
        gradient = {'dW': None, 'db': None}
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate loss and gradient
        # cross = []
        loss = 0
        for i in range(len(y_batch)):
            loss += -(y_batch[i] * math.log(y_pred[i] + 1e-6) - (1 - y_batch[i]) * math.log(1 - y_pred[i] + 1e-6))
        loss = loss / len(y_batch)
        x_batch = (x_batch - np.mean(x_batch)) / np.std(x_batch)
        temp = np.dot(x_batch, self.W) + self.b
        temp = Logistic_Regression.sigmoid(self, temp)
        deriv_W = np.dot(y_batch - temp, np.transpose(x_batch), temp - y_batch)
        deriv_B = y_batch - y_pred

        gradient['dW'] = deriv_W
        gradient['db'] = deriv_B
        # pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return loss, gradient

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE

        z = np.clip(z, -500, 500)
        s = 1 / (1 + np.exp(-z))
        # pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################

        return s

'''
class Naive_Bayes():
    def fit(self, X_train, y_train):
        """
        fit with training data
        Inputs:
            - X_train: A numpy array of shape (N, D) containing training data; there are N
                training samples each of dimension D.
            - y_train: A numpy array of shape (N,) containing training labels; y[i] = c
                means that X[i] has label 0 <= c < C for C classes.
                
        With the input dataset, function gen_by_class will generate class-wise mean and variance to implement bayes inference.

        Returns:
        None
        
        """

        self.x = X_train
        self.y = y_train
        print(np.shape(self.x))
        print(np.shape(self.y))
        self.gen_by_class()

    def gen_by_class(self):
        """
        With the given input dataset (self.x, self.y), generate 3 dictionaries to calculate class-wise mean and variance of the data.
        - self.x_by_class : A dictionary of numpy arraies with the keys as each class label and values as data with such label.
        - self.mean_by_class : A dictionary of numpy arraies with the keys as each class label and values as mean of the data with such label.
        - self.std_by_class : A dictionary of numpy arraies with the keys as each class label and values as standard deviation of the data with such label.
        - self.y_prior : A numpy array of shape (C,) containing prior probability of each class
        """
        self.x_by_class = dict()
        self.mean_by_class = dict()
        self.std_by_class = dict()
        self.y_prior = None

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Generate dictionaries.
        # hint : to see all unique y labels, you might use np.unique function, e.g., np.unique(self.y)
        class_array = []
        class_mean = []
        class_std = []
        for i in range(len(self.y)):
            self.x_by_class = {self.y[i]: self.x[i]}
            class_array.append(self.x_by_class)

        pass
        for k in range(len(self.y)):
            self.mean_by_class = {self.y[k]: Naive_Bayes.mean(self, self.x[k])}
            class_mean.append(self.mean_by_class)

        for i in range(len(self.y)):
            self.std_by_class = {self.y[i]: Naive_Bayes.std(self, self.x[i])}
            class_std.append(self.std_by_class)
        print(class_std)
        # print (self.mean_by_class)
        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        mean = np.mean(x)
        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return mean

    def std(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate standard deviation of input x, do not use np.std
        var = 0
        mean_val = Naive_Bayes.mean(self, x)
        for i in x:
            var += (i - mean_val) ** 2
        variance = var / (len(x))
        std = np.sqrt(variance)
        pass;
        # END_YOUR_CODE
        ############################################################
        ############################################################
        return std

    def calc_gaussian_dist(self, x, mean, std):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate gaussian probability of input x given mean and std
        calc_1 = np.sqrt(2 * math.pi * np.power(std, 2))
        calc_2 = (-np.power((x - mean), 2)) / (2 * np.power(std, 2))
        calc_3 = np.exp(calc_2)
        calc_4 = 1 / calc_1
        gaussian_val = calc_3 * calc_4

        pass
        return gaussian_val
        # END_YOUR_CODE
        ############################################################
        ############################################################

    def predict(self, x):
        """
        Use the acquired mean and std for each class to predict class for input x.
        Inputs:

        Returns:
        - prediction: Predicted labels for the data in x. prediction is (N, C) dimensional array, for N samples and C classes.
        """

        n = len(x)
        num_class = len(np.unique(self.y))
        prediction = np.zeros((n, num_class))

        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x

        pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

        return prediction
'''

class Spam_Naive_Bayes(object):
    """Implementation of Naive Bayes for Spam detection."""

    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def get_word_counts(self, words):
        """
        Generate a dictionary 'word_counts' 
        Hint: You can use helper function self.clean and self.toeknize.
              self.tokenize(x) can generate a list of words in an email x.

        Inputs:
            -words : list of words that is used in a data sample
        Output:
            -word_counts : contains each word as a key and number of that word is used from input words.
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x
        words = self.tokenize(words)
        word_counts = {}
        count = 0
        for i in words:
            if i in word_counts:
                count += 1
                word_counts[i] = count
            else:
                count = 1
                word_counts[i] = count
        #pass
        # END_YOUR_CODE
        ############################################################
        ############################################################

        return word_counts

    def fit(self, X_train, y_train):
        """
        compute likelihood of all words given a class

        Inputs:
            -X_train : list of emails
            -y_train : list of target label (spam : 1, non-spam : 0)
            
        Variables:
            -self.num_messages : dictionary contains number of data that is spam or not
            -self.word_counts : dictionary counts the number of certain word in class 'spam' and 'ham'.
            -self.class_priors : dictionary of prior probability of class 'spam' and 'ham'.
        Output:
            None
        """
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # calculate naive bayes probability of each class of input x

        self.num_messages = {'ham': 0, 'spam': 0}
        self.word_counts = {'ham': {}, 'spam': {}}
        self.class_priors = {'ham': 0, 'spam': 0}
        spam = 0
        non_spam = 0
        for i in range(len(X_train)):
            word_list = Spam_Naive_Bayes.get_word_counts(self, X_train[i])

        ####################################
        # Get the number of messages
            if y_train[i] == 1:
                spam += 1
            elif y_train[i] == 0:
                non_spam += 1
            self.num_messages['ham'] = non_spam
            self.num_messages['spam'] = spam
        ####################################

            if y_train[i] == 1:
                #count = 0
                for k, v in word_list.items():

                    if k in self.word_counts['spam']:
                        self.word_counts['spam'][k] += v
                    else:
                        self.word_counts['spam'][k] = v

            elif y_train[i] == 0:
                for g, f in word_list.items():
                    if g in self.word_counts['ham']:
                        self.word_counts['ham'][g] += f
                    else:

                        self.word_counts['ham'][g] = f
        self.class_priors['spam'] = spam/(spam+non_spam)
        self.class_priors['ham'] = non_spam/(spam+non_spam)

        # END_YOUR_CODE
        ############################################################
        ############################################################

    def predict(self, X):
        """
        predict that input X is spam of not. 
        Given a set of words {x_i}, for x_i in an email(x), if the likelihood 
        
        p(x_0|spam) * p(x_1|sp
        am) * ... * p(x_n|spam) * y(spam) > p(x_0|ham) * p(x_1|ham) * ... * p(x_n|ham) * y(ham),
        
        then, the email would be spam.




        Inputs:
            -X : list of emails

        Output:
            -result : A numpy array of shape (N,). It should tell rather a mail is spam(1) or not(0).
        """

        result = []
        for x in X:
            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # calculate naive bayes probability of each class of input x
            word_list = self.get_word_counts(x)
            probns_x = 1
            probs_x = 1
            ham_num = self.num_messages['ham']
            spam_num = self.num_messages['spam']
            word_ham = self.word_counts['ham']
            word_spam = self.word_counts['spam']
            for i in word_list:
                if i in self.word_counts['ham']:
                    if i not in self.word_counts['spam']:
                        pns_xy = pns_xy
                        probns_x = probns_x * pns_xy
                        ps_xy = pns_xy
                        probs_x = probs_x * ps_xy
                    elif i in self.word_counts['spam']:
                        pns_xy = word_ham[i] / ham_num
                        probns_x = probns_x * pns_xy
                        ps_xy = word_spam[i] / spam_num
                        probs_x = probs_x * ps_xy

            probns_x = probns_x * self.class_priors['ham']
            probs_x = probs_x * self.class_priors['spam']
            if probns_x >= probs_x:
                result.append(0)
            else:
                result.append(1)

            # pass
            # END_YOUR_CODE
            ############################################################
            ############################################################

        result = np.array(result)
        return result
