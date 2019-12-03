import numpy as np


class NeuralNetwork:
    """Returns a neural network object"""
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """Initializes the neural network
            
            Arguments
            ---------
            
            input_nodes: Integer, number of inputs nodes
            hidden_nodes: Integer, number of hidden nodes
            output_nodes: Interger, number of output nodes
 
 
        """
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        
        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Define activation function
        self.activation_function = lambda x : (1/(1+np.exp(-x)))
        self.derivative_sigmoid = lambda x : (x * (1 - x))
        
        
    def train(self, features, targets):
        """Trains the neural network on batch of features and targets
        
           Arguments
           ---------
           
           features: 2D array, each row is one data record, each column is a feature
           targets: 1D array, target values
           
        """ 
        n_records = features.shape[0]
        delta_weights_input_to_hidden = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_hidden_to_output = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            (delta_weights_input_to_hidden, 
             delta_weights_hidden_to_output) = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                    delta_weights_input_to_hidden,
                                                                    delta_weights_hidden_to_output)
        self.update_weights(delta_weights_input_to_hidden, delta_weights_hidden_to_output, n_records)
        
        
    def forward_pass_train(self, X):
        """Forward pass of the features through the network
        
            Arguments
            ---------
            X: features that are being passed through the network
            
        """
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # Final outputs = final inputs, due to not using an activation function
        # on the output layer, as this is a regression approach
        # Alternatively use the line below:
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs
    
    
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_input_to_hidden, 
                        delta_weights_hidden_to_output):
        """Implementing backpropagation to adjust weights
        
            Arguments
            ---------
            final_outputs: Values of the output layer from the forward pass
            hidden_outputs: Values of the hidden layer from the forward pass
            X: Features
            y: Target values
            delta_weights_input_to_hidden: Weights between input and hidden layer
            delta_weights_hidden_to_output: Weights between hidden and output layer
            
        """
        
        error = y - final_outputs
        hidden_error = error * self.weights_hidden_to_output
        
        # Output error term here is simply the error, because the activation function = x
        # Hence the derivative of the activation function = 1
        output_error_term = error
        
        hidden_error_term = hidden_error * self.derivative_sigmoid(hidden_outputs).reshape(-1,1)
        hidden_error_term = hidden_error_term.transpose()
        
        # Weight step (input to hidden)
        delta_weights_input_to_hidden += self.lr * np.dot(X.reshape(-1, 1), hidden_error_term)
        
        # Weight step (hidden to output)
        delta_weights_hidden_to_output += self.lr * np.multiply(hidden_outputs.reshape(-1,1), output_error_term)
        
        return delta_weights_input_to_hidden, delta_weights_hidden_to_output
        
        
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records
        '''
        
        self.weights_hidden_to_output += delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step

        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = self.activation_function(final_inputs) # signals from final output layer         
        final_outputs = final_inputs
        
        return final_outputs


class NeuralNet:
    def __init__(self):
        pass


    def forward_pass(self, X):
        # for layer in network:
            # calculate output from layer
            # calculate activation output
            # take result as input for next iteration
            # take next weight-set
            # iterate again
        # return output value(s) of last layer
        # Is it necessary to also return the output values of the hidden layers?
        pass


    def output_from_layer(self, inputs, weights):
        return np.dot(inputs, weights)
    
#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
    
        
        
        
        
        