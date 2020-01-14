#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

#%%
class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()
        
    def init_distribution(self, arms):
        # Num states = Num Arms to keep things simple
        self.bandit_matrix = np.random.rand(arms, arms)
        #each row represents a state, each column an arm
        
    def reward(self, prob):
        reward = 0
        for _ in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward
        
    def get_state(self):
        return self.state
    
    def update_state(self):
        self.state = np.random.randint(0, self.arms)
        
    def get_reward(self,arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])
        
    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward

def softmax(av, tau = 1.12):
   n = len(av)
   probs = np.zeros(n)
   for i in range(n):
       softm = (np.exp(av[i] / tau) / np.sum(np.exp(av[:] / tau)))
       probs[i] = softm
   return probs

def one_hot(N, pos, val=1):
   one_hot_vec = np.zeros(N)
   one_hot_vec[pos] = val
   return one_hot_vec


arms = 10
model_optimizer = Adam(1e-2)
epochs = 5000

#%%
input_layer = Input(shape=(arms))
x = Dense(100, activation="relu")(input_layer)
output_layer = Dense(arms, activation="relu")(x)

model = Model(input_layer, output_layer)
model.summary()

env = ContextBandit(arms)
reward_hist = np.zeros(50)

#%%
def step(current_state, i, running_mean):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(current_state)                                       #produce reward predictions
        av_softmax = softmax(y_pred.numpy().ravel(), tau=2.0)               #turn reward distribution into probability distribution
        av_softmax /= av_softmax.sum()                                      #make sure total prob adds to 1
        choice = np.random.choice(arms, p=av_softmax)                       #sample an action
        cur_reward = env.choose_arm(choice)
        one_hot_reward = y_pred.numpy().ravel().copy()
        one_hot_reward[choice] = cur_reward
        reward = tf.Variable(tf.convert_to_tensor(one_hot_reward))
        model_loss = tf.keras.losses.mean_squared_error(y_pred, reward)     
        if i % 50 == 0:
            running_mean = np.average(reward_hist)
            reward_hist[:] = 0
            plt.scatter(i, running_mean)
        reward_hist[i % 50] = cur_reward

    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))


#%%
def train(env):
    #one-hot encode current state
    cur_state = tf.Variable(tf.convert_to_tensor(one_hot(arms, env.get_state()).reshape(-1,10)))
    reward_hist[:] = 5
    running_mean = np.average(reward_hist)

    plt.xlabel("Plays")
    plt.ylabel("Mean Reward")

    for i in range(epochs):
        step(cur_state, i, running_mean)
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=model_optimizer)
        cur_state = tf.Variable(tf.convert_to_tensor(one_hot(arms, env.get_state()).reshape(-1,10)))

train(env)

