import numpy as np
import random

from keras import initializers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from environment.environment import sample_action_customer
from params import EPSILON_START, DISCOUNT_RATE, EPSILON_MIN, EPSILON_DECAY, CUSTOMER_ACTION_SIZE, BUFFER_SIZE, \
    BATCH_SIZE, CUSTOMER_STATE_SIZE, TRAINING_INTERVAL, \
    REPLACE_TARGET_INTERVAL, TAU, LEARNING_RATE_DQN, HIDDEN_LAYER_SIZE, POWER_RATES
from utils.replay_buffer import ReplayBuffer


def construct_network():
    """ Construct the Deep-Q network. It consists of an input layer with the size of the state variables, 2 hidden
    layers and an output layer with the size of the possible actions. """
    model = Sequential()
    model.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(CUSTOMER_STATE_SIZE,), activation='relu',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.add(Dense(CUSTOMER_ACTION_SIZE, activation='linear',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE_DQN))
    return model


def predict(state, network):
    """ Predict the Q-values for a given state and network. """
    state_input = np.reshape(state, (-1, CUSTOMER_STATE_SIZE))
    return network(state_input)


class CustomerAgent:
    """ This CustomerAgent is a Reinforcement Learning agent using a Deep-Q network to predict the Q-values of
    state-action pairs. In the act function the agent calls for the previous reward and the next observation.
    It updates its network based on the previous reward, observation and action. Then it decides upon the next action.
    """

    def __init__(self, agent_id, data_id, env, dummy=False, q_network=None, target_network=None):
        self.agent_id = agent_id
        self.data_id = data_id
        self.env = env
        self.epsilon = EPSILON_START
        self.dummy = dummy
        self.acc_reward = 0
        self.last_state = None
        self.last_action = None
        self.last_history = None
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.visited = {}
        self.q_network = q_network
        self.target_network = target_network
        if q_network is None:
            self.q_network = construct_network()
        if target_network is None:
            self.target_network = construct_network()

    def reset(self):
        """ Reset the agent before each episode. """
        self.last_state = None
        self.last_action = None
        self.last_history = None
        self.acc_reward = 0
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def act(self, train=True):
        """ Select an action based on the observation. If the agent is in training and is not a dummy the agent's
        Q-network is also updated in the step function. """
        observation, reward, done, _ = self.env.last_customer(self.agent_id)

        if train and not self.dummy:
            if self.last_action is not None:
                self.step(self.last_state, self.last_action, reward, observation, done)
            action = self.choose_action(observation, self.epsilon)
        else:
            action = self.choose_action(observation)

        self.env.act(self.agent_id, action)
        self.last_state = observation
        self.last_action = action
        self.acc_reward += reward

    def choose_action(self, s, eps=0.0):
        """ Choose an action based on the given state. If the agent is a dummy it will simply consume an amount equal
        to its demand (power rate 1.0). Otherwise an action is selected based on epsilon-greedy. """
        if self.dummy:
            return POWER_RATES.index(1.0)
        elif random.uniform(0, 1) < eps:
            return sample_action_customer()
        else:
            actions = predict(s, self.q_network)
            action = np.argmax(actions)
            return action

    def step(self, state, action, reward, next_state, done):
        """ Every iteration the agent takes a training step. The agent adds the SARS tuple to the replay buffer. The
        replay buffer is then used for sampling a batch for training. """
        self.memory.add(state, action, reward, next_state, done)

        # Train network on a certain interval and if the replay buffer has enough samples
        if len(self.memory) >= BATCH_SIZE and self.env.curr_step % TRAINING_INTERVAL == 0:
            sampled_experiences = self.memory.sample()
            self.train(sampled_experiences)

        # Replace target network on a certain interval
        if self.env.episode % REPLACE_TARGET_INTERVAL == 0:
            self.target_network.set_weights(self.q_network.get_weights())

    def train(self, experiences):
        """ Train the Q-network. The target values are based on a target network to stabilize training. """
        states, actions, rewards, next_states, dones = experiences
        outputs = predict(next_states, self.target_network)
        next_actions = np.max(outputs, axis=1)
        target_values = rewards + (DISCOUNT_RATE * next_actions * (1 - dones))
        targets = predict(states, self.q_network).numpy()
        targets[np.arange(len(states)), actions] = target_values
        self.last_history = self.q_network.fit(np.array(states), np.array(targets), verbose=False)

    def update_network(self):
        """ Do a soft update on the target network. A soft update can be done every iteration. This is slightly
        different from a hard update on an interval. This is currently not used. """
        model_weights = self.q_network.get_weights()
        target_model_weights = self.target_network.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_network.set_weights(target_model_weights)

    def save(self, path):
        """ Save the network. """
        self.q_network.save(path + '/Q_network_' + str(self.data_id) + '.h5')
        np.save(path + '/dissatisfaction_coefficients_' + str(self.data_id) + '.npy', self.env.dissatisfaction_coefficients[self.agent_id])
        print("Successfully saved network for agent " + str(self.data_id))

    def load(self, path):
        """ Load a network give its path. """
        self.q_network = load_model(path + '/Q_network_' + str(self.data_id) + '.h5')
        self.env.dissatisfaction_coefficients[self.agent_id] = np.load(path + '/dissatisfaction_coefficients_' + str(self.data_id) + '.npy')
