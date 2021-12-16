import random
from gym import Env as OpenAIEnv
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_INT = 2147483647
MAX_STEPS = 60000


class Actions:
    Buy = 0
    Sell = 1
    Hold = 2
    N = 3


class SmartBrokerEnv(OpenAIEnv):
    def __init__(
        self,
        df_info,
        portfolio,
        batch_dur=30,
        n_actions=Actions.N,
        data_dir='../data',
    ):
        self.df_info = df_info
        self.reward_range = (0, MAX_INT)
        self.batch_dur = batch_dur
        self.df_info = df_info
        self.portfolio = portfolio
        self.n_actions = n_actions
        self.curr_step = 0
        self.data_dir = data_dir
        self._init_portfolio(load_df=True)
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([n_actions, 1]),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.batch_dur*4 + 3, 1),
            dtype=np.uint8,
        )

    def _init_portfolio(self, load_df=False):
        self.source = self.portfolio.get('source', 'Bitstamp')
        self.init_balance = self.portfolio.get('init_balance', 100)
        self.balance = self.init_balance
        self.entity = self.portfolio.get('entity', 'XRP')
        self.year = self.portfolio.get('year', '2021')
        self.market = self.portfolio.get('market', 'USD')
        self.duration_typ = self.portfolio.get('duration_typ', 'minute')
        self.price_typ = self.portfolio.get('price_typ', 'close')
        self.roll_period = self.portfolio.get('roll_period', 30)
        self.start_dt = self.df_info.get('start_date')
        self.end_dt = self.df_info.get('end_date')
        self.units_held = 0
        self.net_worth = self.balance

        if load_df:
            file = f'{self.source}_{self.entity}{self.market}_{self.year}_{self.duration_typ}.csv'
            self.df = pd.read_csv(f'{self.data_dir}/{file}', skiprows=1)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.sort_values(by='date', inplace=True, ascending=True)
            self.df.reset_index(inplace=True)
            self.df = self.df[self.df_info.get('cols')]
            # process and initialise dataframe
            self._process_df()

        self.max_step = self.df.shape[0]

    def _process_df(self):
        norm_cols = self.df_info.get('norm_cols')

        self.df[norm_cols] = self.df[norm_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()),
        )
        self.df['rolling_price'] = self.df[self.price_typ].rolling(self.roll_period).sum()
        self.df['rsi'] = self._rsi(self.df[self.price_typ])
        self.df['bolu'], self.df['bold'] = self._bollingerbands()
        self.df_main = self.df.copy()
        # filter based on range
        # self.df = self.df.loc[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]

    def _get_ptfo_ftrs(self):
        # normalise features
        return np.array([self.balance/MAX_INT, self.units_held/MAX_INT, self.net_worth/MAX_INT])

    def _get_obs(self):
        prices = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ][self.price_typ].values 

        roll_prices = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ]['rolling_price'].values 

        volumes = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ][f'Volume {self.entity}'].values 

        rsi = self.df.iloc[
            self.curr_step: self.curr_step + self.batch_dur
        ][f'rsi'].values 

        ptfo_ftrs = self._get_ptfo_ftrs()

        BOLU = self.df.iloc[self.curr_step: self.curr_step + self.batch_dur
        ][f'bolu'].values

        BOLD = self.df.iloc[self.curr_step: self.curr_step + self.batch_dur
        ][f'bold'].values

        obs = np.concatenate(
            (
                prices,
                roll_prices,
                volumes,
                rsi,
                ptfo_ftrs,
                BOLU,
                BOLD,
            )
        )

        return obs

    def _rsi(self, prices, com=13):
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=com, adjust=False).mean()
        ema_down = down.ewm(com=com, adjust=False).mean()
        rs = ema_up/ema_down
        return rs

    def _bollingerbands(self):
        df=self.df
        df['TP'] = (df['close'] + df['low'] + df['high'])/3
        df['std'] = df['TP'].rolling(self.roll_period).std(ddof=0)
        df['MA-TP'] = df['TP'].rolling(self.roll_period).mean()
        df['bolu'] = df['MA-TP'] + 2*df['std']
        df['bold'] = df['MA-TP'] - 2*df['std']
        return df['bolu'].values,df['bold'].values

    def _act(self, action):
        # default to selling or buying all the stocks
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action_type = abs(action[0])
            amount = abs(action[1])
        else:
            action_type = action
            amount = 1

        curr_price = self.df.iloc[self.curr_step][self.price_typ]
        units_bought = 0
        units_sold = 0
        alpha = self.curr_step / MAX_STEPS

        if action_type < 1:
            action_type = int(action_type * self.action_space.high[0])

        if action_type == Actions.Buy:
            total_possible = int(self.balance / curr_price)
            units_bought = int(total_possible * amount)
            cost = units_bought * curr_price
            self.balance -= cost
            self.units_held += units_bought
        elif action_type == Actions.Sell:
            units_sold = self.units_held * amount
            self.balance += units_sold * curr_price
            self.units_held -= units_sold

#         self.net_worth = self.balance + self.units_held * curr_price

        if action_type == Actions.Buy and total_possible == 0:
            reward = -10
        elif action_type == Actions.Sell and self.units_held == 0:
            reward = -5
        else:
            if(self.net_worth>100):
                reward = self.units_held * curr_price - self.init_balance +5
            else:
                reward = self.units_held * curr_price - self.init_balance
        self.net_worth = self.balance + self.units_held * curr_price
        info = {
            'amount': amount,
            'reward': reward,
            'curr_price': curr_price,
            'action': action_type,
            'curr_step': self.curr_step,
            'units_bought': units_bought,
            'units_sold': units_sold,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'units_held': self.units_held,
            'profit': self.net_worth - self.init_balance,
        }

        return info

    def reset(self, idx=None, randomize=False):
        if idx is None:
            idx = self.roll_period
        self._init_portfolio()
        if not randomize:
            self.curr_step = idx
        else:
            self.curr_step = random.randint(idx, self.df[self.df.date == self.end_dt].index[0])
        obs = self._get_obs()
        return obs

    def step(self, action):
        info = self._act(action)
        self.curr_step += 1
        reward = info['reward']
        done = self.net_worth <= 0 or self.curr_step == MAX_STEPS
        obs = self._get_obs()

        if self.curr_step > self.df.shape[0] - self.batch_dur:
            self.curr_step = random.randint(self.roll_period, self.df.shape[0] - self.batch_dur)

        return obs, reward, done, info

    def render(self, *args):
        buy_s, buy_p, sell_s, sell_p, pred_buy_p, pred_buy_s, pred_sell_p, pred_sell_s, start_step, end_step, show_logs, show_pred = args

        if show_logs:
            print(f'Buy Steps: {buy_s}')
            print(f'Sell Steps: {sell_s}')

        start_step = max(self.roll_period, start_step-5)
        end_step = min(self.df.shape[0], end_step+5)
        df = self.df_main.loc[start_step:end_step]
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))

        ax.plot(df['date'], df['close'], color='black', label='XRP')
        ax.scatter(df.loc[buy_s, 'date'].values, buy_p, c='green', alpha=0.5, label='buy')
        ax.scatter(df.loc[sell_s, 'date'].values, sell_p, c='red', alpha=0.5, label='sell')
        if show_pred:
            ax.scatter(df.loc[pred_buy_s, 'date'].values, pred_buy_p, c='blue', alpha=0.5, label='predicted buy')
            ax.scatter(df.loc[pred_sell_s, 'date'].values, pred_sell_p, c='orange', alpha=0.5, label='predicted sell')

        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        plt.show()

    def close(self):
        print('close')