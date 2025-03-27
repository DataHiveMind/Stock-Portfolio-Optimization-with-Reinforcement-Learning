# src/rl_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import random
import sqlite3

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = np.zeros(len(data.columns))
        self.current_step = 0
        self.action_space = np.array([0, 1, 2]) # 0: hold, 1: buy, 2: sell

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio = np.zeros(len(self.data.columns))
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.data.iloc[self.current_step].values, self.portfolio, [self.balance]])

    def step(self, action):
        prices = self.data.iloc[self.current_step].values
        if action == 1: # Buy
            for i in range(len(prices)):
                if self.balance > prices[i]:
                    shares = self.balance // prices[i]
                    self.balance -= shares * prices[i]
                    self.portfolio[i] += shares
        elif action == 2: # Sell
            for i in range(len(prices)):
                self.balance += self.portfolio[i] * prices[i]
                self.portfolio[i] = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.balance - self.initial_balance if done else 0
        return self._get_state(), reward, done, {}

def build_model(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_size,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_dqn(env, model, episodes=1000):
    memory = []
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = np.argmax(model.predict(np.array([state])))
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                targets = model.predict(np.array(states))
                next_q_values = model.predict(np.array(next_states))
                for i in range(batch_size):
                    if dones[i]:
                        targets[i][actions[i]] = rewards[i]
                    else:
                        targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
                model.train_on_batch(np.array(states), targets)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode: {episode + 1}, Balance: {env.balance}")
    return model

if __name__ == "__main__":
    conn = sqlite3.connect('data/database/stock_data.db')
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)

    env = TradingEnvironment(df)
    state_size = len(env._get_state())
    action_size = len(env.action_space)
    model = build_model(state_size, action_size)
    trained_model = train_dqn(env, model)
    trained_model.save("rl_model.h5")