import numpy as np
import random

from keras import initializers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from environment.environment import sample_action_aggregator
from params import EPSILON_START, EPSILON_MIN, EPSILON_DECAY, \
    BUFFER_SIZE, BATCH_SIZE, TRAINING_INTERVAL, REPLACE_TARGET_INTERVAL, \
    TAU, LEARNING_RATE_DQN, HIDDEN_LAYER_SIZE, AGGREGATOR_ACTION_SIZE, AGGREGATOR_STATE_SIZE, DISCOUNT_RATE_AGGREGATOR
from utils.replay_buffer import ReplayBuffer


def construct_network():
    model = Sequential()
    model.add(Dense(HIDDEN_LAYER_SIZE, input_shape=(AGGREGATOR_STATE_SIZE,), activation='relu',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.add(Dense(HIDDEN_LAYER_SIZE, activation='relu',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.add(Dense(AGGREGATOR_ACTION_SIZE, activation='linear',
                    kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE_DQN))
    return model


def predict(state, network):
    state_input = np.reshape(state, (-1, AGGREGATOR_STATE_SIZE))
    return network(state_input)


class AggregatorAgent:
    """ This AggregatorAgent is a deep Reinforcement Learning agent similar to the CustomerAgent. Currently the
    Q-learning AggregatorAgent is used. """

    def __init__(self, env):
        self.env = env
        self.epsilon = EPSILON_START
        self.acc_reward = 0
        self.last_state = None
        self.last_action = None
        self.last_history = None
        self.q_network = construct_network()
        self.target_network = construct_network()
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_history = None
        self.acc_reward = 0
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def act(self, train=True):
        observation, reward, done, _ = self.env.last_aggregator()

        if train:
            if self.last_action is not None:
                self.step(self.last_state, self.last_action, reward, observation, done)
            action = self.choose_action(observation, self.epsilon)
        else:
            action = self.choose_action(observation)

        self.env.act_aggregator(action)
        self.last_state = observation
        self.last_action = action
        self.acc_reward += reward

    def choose_action(self, s, eps=0.0):
        if random.uniform(0, 1) < eps:
            return sample_action_aggregator()
        else:
            actions = predict(s, self.q_network)
            action = np.argmax(actions)
            return action

    def step(self, state, action, reward, next_state, done, name=None):
        self.memory.add(state, action, reward, next_state, done)

        # Train network
        if len(self.memory) >= BATCH_SIZE and self.env.curr_step % TRAINING_INTERVAL == 0:
            sampled_experiences = self.memory.sample()
            self.train(sampled_experiences)

        # Replace target network
        if self.env.episode % REPLACE_TARGET_INTERVAL == 0:
            self.target_network.set_weights(self.q_network.get_weights())

    def train(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        outputs = predict(next_states, self.target_network)
        next_actions = np.max(outputs, axis=1)
        target_values = rewards + (DISCOUNT_RATE_AGGREGATOR * next_actions * (1 - dones))
        targets = predict(states, self.q_network).numpy()
        targets[np.arange(len(states)), actions] = target_values
        self.last_history = self.q_network.fit(np.array(states), np.array(targets), verbose=False)

    def update_network(self):
        model_weights = self.q_network.get_weights()
        target_model_weights = self.target_network.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_network.set_weights(target_model_weights)

    def save(self, path):
        self.q_network.save(path + '/Q_network_aggregator.h5')
        print("Successfully saved network.")

    def load(self, path):
        self.q_network = load_model(path + '/Q_network_aggregator.h5')
