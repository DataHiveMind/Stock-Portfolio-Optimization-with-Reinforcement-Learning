# src/rl_model/environment.py
import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = np.zeros(len(data.columns))
        self.current_step = 0
        self.action_space = np.array([0, 1, 2])  # 0: hold, 1: buy, 2: sell

    def reset(self):
        self.balance = self.initial_balance
        self.portfolio = np.zeros(len(self.data.columns))
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.data.iloc[self.current_step].values, self.portfolio, [self.balance]])

    def step(self, action):
        prices = self.data.iloc[self.current_step].values
        if action == 1:  # Buy
            for i in range(len(prices)):
                if self.balance > prices[i]:
                    shares = self.balance // prices[i]
                    self.balance -= shares * prices[i]
                    self.portfolio[i] += shares
        elif action == 2:  # Sell
            for i in range(len(prices)):
                self.balance += self.portfolio[i] * prices[i]
                self.portfolio[i] = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.balance - self.initial_balance if done else 0
        return self._get_state(), reward, done, {}