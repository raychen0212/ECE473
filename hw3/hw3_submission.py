import collections
import math
import numpy as np

class Gaussian_Naive_Bayes():
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
        y_label = np.unique(self.y)

        for i in range(len(y_label)):
            self.x_by_class[y_label[i]] = []
            self.mean_by_class[y_label[i]] = []
            self.std_by_class[y_label[i]] = []
        for i in range(len(self.x)):
            self.x_by_class[self.y[i]].append(self.x[i])

        for k,a in self.x_by_class.items():
            self.mean_by_class[k].append(self.mean(a))

        for k,a in self.x_by_class.items():
            self.std_by_class[k].append(self.std(a))


        self.y_prior = np.zeros(len(y_label))
        for k in y_label:
            for i in range(len(self.y)):
                if self.y[i] == k:
                    self.y_prior[k] += 1
        self.y_prior = self.y_prior / len(self.y)

        # END_YOUR_CODE
        ############################################################
        ############################################################        

    def mean(self, x):
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Calculate mean of input x
        sum = 0

        for i in x:
            sum += i
        mean = sum/(len(x)+1)
        pass;
    
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
        mean_val = self.mean(x)
        for i in range(len(x)):
            var += (x[i] - mean_val) ** 2
        variance = var / (len(x)+1)
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
        if std == 0:
            gaussian_val = 1
        else:
            calc_1 = std*np.sqrt(2 * math.pi)
            calc_2 = (((x-mean)/(std)) ** 2) * (-1/2)
            calc_3 = np.exp(calc_2)
            calc_4 = 1 / (calc_1)
            gaussian_val = np.log(calc_3 * calc_4+0.001)
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

        for k, a in self.mean_by_class.items():
            for i in range(len(x)):
                gau_value = 0
                for j in range(len(x[1])):
                    #print(self.std_by_class)
                    gau_value += self.calc_gaussian_dist(x[i][j],a[0][j],self.std_by_class[k][0][j])
                prediction[i][k]=gau_value + self.y_prior[k]
                #print(prediction)
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return prediction


class Neural_Network():
    def __init__(self, hidden_size = 64, output_size = 1):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None    
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def fit(self, x, y, batch_size = 64, iteration = 2000, learning_rate = 1e-3):
        """
        Train this 2 layered neural network classifier using mini-batch stochastic gradient descent.
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

        #initialize W
        if self.W1 == None:
            self.W1 = 0.001 * np.random.randn(dim, self.hidden_size)
            self.b1 = 0
            
            self.W2 = 0.001 * np.random.randn(self.hidden_size, self.output_size)
            self.b2 = 0


        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]

            loss, gradient = self.loss(x_batch, y_batch)

            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Update parameters with mini-batch stochastic gradient descent method
            for i in range(len(gradient['dW2'])):
                self.W2[i] -= learning_rate * gradient['dW2'][i]
            for b in range(len(gradient['db2'])):
                self.b2 -= learning_rate * gradient['db2'][b]
            for i in range(len(gradient['dW1'])):
                self.W1[i] -= learning_rate * gradient['dW1'][i]
            for b in range(len(gradient['db1'])):
                self.b1 -= learning_rate * gradient['db1'][b]
            #print("dW1", np.shape(gradient['dW1']), "dW2", np.shape(gradient['dW2']), "db1", np.shape(gradient['db1']),
                  #"db2", np.shape(gradient['db2']))
            pass;
        
            # END_YOUR_CODE
            ############################################################
            ############################################################
            
            y_pred = self.predict(x_batch)
            acc = np.mean(y_pred == y_batch)
            
            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))
                
    def loss(self, x_batch, y_batch, reg = 1e-3):
            """
            Implement feed-forward computation to calculate the loss function.
            And then compute corresponding back-propagation to get the derivatives. 

            Inputs:
            - X_batch: A numpy array of shape (N, D) containing a minibatch of N
              data points; each point has dimension D.
            - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
            - reg: hyperparameter which is weight of the regularizer.

            Returns: A tuple containing:
            - loss as a single float
            - gradient dictionary with four keys : 'dW1', 'db1', 'dW2', and 'db2'
            """
            gradient = {'dW1' : None, 'db1' : None, 'dW2' : None, 'db2' : None}


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate y_hat which is probability of the instance is y = 0.
            x_batch = (x_batch - np.mean(x_batch)) / np.std(x_batch)
            y_calc = np.dot(x_batch, self.W1) + self.b1
            h1 = self.activation(y_calc)
            g2 = np.dot(h1,self.W2) + self.b2
            y_pred = self.sigmoid(g2)
            pass;

            # END_YOUR_CODE
            ############################################################
            ############################################################


            ############################################################
            ############################################################
            # BEGIN_YOUR_CODE
            # Calculate loss and gradient

            loss = 0
            for i in range(len(y_batch)):
                loss += (-y_batch[i] * math.log(y_pred[i] + 1e-6) - (1 - y_batch[i]) * math.log(1 - y_pred[i] + 1e-6))
                loss += reg * (np.sum(self.W2 ** 2) + np.sum(self.W1 ** 2) + np.sum(self.b1 ** 2) + np.sum(self.b2 ** 2))
            loss = loss / len(y_batch)

            #deriv_W2 = np.dot(np.transpose(x_batch), y_pred - y_batch)

            deriv_W2 = np.dot(h1,y_pred - y_batch)+2*reg*self.W2
            deriv_B2 = y_batch - y_pred
            deriv_W1 = np.dot(np.transpose(x_batch),(np.dot((y_pred - y_batch),(np.transpose(self.W2)))))+2*reg*self.W1
            deriv_B1 = np.dot(np.transpose(self.W2),y_pred-y_batch)

            gradient['dW2'] = deriv_W2
            gradient['db2'] = deriv_B2
            gradient['dW1'] = deriv_W1
            gradient['db1'] = deriv_B1

            # END_YOUR_CODE
            ############################################################
            ############################################################
            return loss, gradient

    def activation(self, z):
        """
        Compute the ReLU output of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : output of ReLU(z)
        """ 
        ############################################################
        ############################################################
        # BEGIN_YOUR_CODE
        # Implement ReLU

        for a in range(len(z)):
            for i in range(len(z[a])):
                if i >= 0:
                    z[a][i] = z[a][i]
                else:
                    z[a][i] = 0
            pass;
        s = z
        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s
        
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
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        
        return s
    
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
        x = (x-np.mean(x))/np.std(x)
        y_calc = np.dot(x, self.W1) + self.b1
        h1 = self.activation(y_calc)
        g2 = np.dot(h1, self.W2) + self.b2
        y_hat = self.sigmoid(g2)
        for i in range(len(y_hat)):
            if y_hat[i] >= 0.5:
                y_hat[i] = 1
            elif y_hat[i] < 0.5:
                y_hat[i] = 0
        pass;

        # END_YOUR_CODE
        ############################################################
        ############################################################
        return y_hat

    