from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

class Actor:
    '''
    Actor (Policy) Model.
    
    Model: policy function mu(s | theta^{mu}).
        inputs: states values
        outputs: predicted actions.values
    Optimization: gradient descent
    '''

    def __init__(self, state_size, action_size, action_low, action_high, learning_rate):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            learning_rate 
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.learning_rate = learning_rate
        # model
        self.model = None
        # function for policy update
        self.train_fn = None
        # create model
        self.build_model()

    def build_model(self):
        '''Build an actor (policy) network.'''

        ########################### Network ###########################
        # input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # hidden1
        h1 = layers.Dense(units=32, activation=None, use_bias=False)(states)
        h1 = layers.BatchNormalization()(h1)
        h1 = layers.Activation('relu')(h1)
        h1 = layers.Dropout(0.5)(h1)
        
        # hidden2
        h2 = layers.Dense(units=64, activation=None, use_bias=False)(h1)
        h2 = layers.BatchNormalization()(h2)
        h2 = layers.Activation('relu')(h2)
        h2 = layers.Dropout(0.5)(h2)
        
        # hidden3
        h3 = layers.Dense(units=32, activation=None, use_bias=False)(h2)
        h3 = layers.BatchNormalization()(h3)
        h3 = layers.Activation('relu')(h3)
        h3 = layers.Dropout(0.5)(h3)
        
        # output layer with sigmoid activation; output range -> [0,1]
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                   name='raw_actions')(h3)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        ##################### Loss & Optimization ######################
        
        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    '''
    Critic (Value) Model.
    
    Model: q_hat(s,a,w).
        inputs: states and action values
        output: predicted Q-values of the inputs
    Optimization:
        target: better estimated reward of the greedy-policy
            R_t+1 + gamma * Q(s_t+1, policy(s_t+1))
        loss: estimated reward
            mean squared error of "(target) - (predicted Q-values of the inputs)"
    '''

    def __init__(self, state_size, action_size, learning_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            learning_rate:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        # model
        self.model = None
        # action gradient for policy update
        self.get_action_gradients = None
        # create model
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        ########################### Network ###########################
        # input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # hidden layer for state pathway
        h1_states = layers.Dense(units=32, activation=None, use_bias=False)(states)
        h1_states = layers.BatchNormalization()(h1_states)
        h1_states = layers.Activation('relu')(h1_states)
        h1_states = layers.Dropout(0.5)(h1_states)
        
        h2_states = layers.Dense(units=64, activation=None, use_bias=False)(h1_states)
        h2_states = layers.BatchNormalization()(h2_states)
        h2_states = layers.Activation('relu')(h2_states)
        h2_states = layers.Dropout(0.5)(h2_states)
        
        h3_states = layers.Dense(units=32, activation=None, use_bias=False)(h2_states)
        h3_states = layers.BatchNormalization()(h3_states)
        h3_states = layers.Activation('relu')(h3_states)
        h3_states = layers.Dropout(0.5)(h3_states)
        
        # Add hidden layer(s) for action pathway
        h1_actions = layers.Dense(units=32, activation=None, use_bias=False)(actions)
        h1_actions = layers.BatchNormalization()(h1_actions)
        h1_actions = layers.Activation('relu')(h1_actions)
        h1_states = layers.Dropout(0.5)(h1_states)
        
        h2_actions = layers.Dense(units=64, activation=None, use_bias=False)(h1_actions)
        h2_actions = layers.BatchNormalization()(h2_actions)
        h2_actions = layers.Activation('relu')(h2_actions)
        h2_states = layers.Dropout(0.5)(h2_states)
        
        h3_actions = layers.Dense(units=32, activation=None, use_bias=False)(h2_actions)
        h3_actions = layers.BatchNormalization()(h3_actions)
        h3_actions = layers.Activation('relu')(h3_actions)
        h3_states = layers.Dropout(0.5)(h3_states)
        
        # Combine state and action pathways
        add_net = layers.Add()([h3_states, h3_actions])
        add_net = layers.Activation('relu')(add_net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(add_net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        ###################### Loss & Optimization ######################
        
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        ########## gradient caluculation for the actor network ##########
        
        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
