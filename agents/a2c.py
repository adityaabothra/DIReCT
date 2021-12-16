from collections import defaultdict, deque

from scipy import stats
import numpy as np
import torch
from torch import FloatTensor as FT, tensor as T


class A2C:
    def __init__(
        self,
        env,
        actor,
        critic,
        n_actns, 
        actor_optmz, 
        critic_optmz,
        mdl_pth,
        log_freq=100,
        hyprprms={},
        p_net_type='nn',
        c_net_type='nn',
        load_models=False,
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.n_actns = n_actns
        self.actor_optmz = actor_optmz
        self.critic_optmz = critic_optmz
        self.log_freq = log_freq
        self.mdl_pth = mdl_pth
        self.hyprprms = hyprprms
        self.gamma = self.hyprprms.get('gamma', 0.95),
        self.step_sz = self.hyprprms.get('step_sz', 0.001)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.p_net_type = p_net_type
        self.c_net_type = c_net_type
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )
        self.eval_logs = {}
        self.load_models = load_models
        self.curr_step = 0

        if self.p_net_type == 'lstm':
            self.p_hdn_st = self.actor.init_states(1)

        if self.c_net_type == 'lstm':
            self.c_hdn_st = self.critic.init_states(1)

        if self.load_models:
            self.actor.load_state_dict(torch.load(f'{mdl_pth}/actor'))
            self.critic.load_state_dict(torch.load(f'{mdl_pth}/critic'))

    @staticmethod
    def _normalise(arr):
        mean = arr.mean()
        std = arr.std()
        arr -= mean
        arr /= (std + 1e-5)
        return arr

    def _get_returns(self, trmnl_state_val, rewards, gamma=1, normalise=True):
        R = trmnl_state_val
        returns = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R 
            returns.append(R)

        returns = returns[::-1]
        if normalise:
            return self._normalise(torch.cat(returns))

        return FT(returns)

    def _get_action(self, policy):
        actn = T(policy.sample().item())
        actn_log_prob = policy.log_prob(actn).unsqueeze(0)
        return actn, actn_log_prob

    def train(self):
        exp = []
        state = self.env.reset()
        ts = 0
        ep_ended = False
        ep_reward = 0
        ep_loss = 0
        net_worth = 0
        profit = 0
        bal = 0
        units_held = 0
        state = FT(state)

        while not ep_ended:
            if self.p_net_type == 'lstm':
                policy, self.p_hdn_st = self.actor.forward(
                    state,
                    self.p_hdn_st,
                )
            else:
                policy = self.actor(state)

            actn, actn_log_prob = self._get_action(policy)

            if self.c_net_type == 'lstm':
                state_val, self.c_hdn_st = self.critic.forward(
                    state,
                    self.c_hdn_st,
                )
            else:
                state_val = self.critic(state)

            nxt_state, reward, ep_ended, info = self.env.step(action=actn.item())
            nxt_state = FT(nxt_state)
            exp.append((nxt_state, state_val, T([reward]), actn_log_prob))
            ep_reward += info.get('reward')
            profit += info.get('profit')
            bal += info.get('balance')
            units_held += info.get('units_held')
            net_worth += info.get('net_worth')
            state = nxt_state
            ts += 1
            self.curr_step += 1

        states, state_vals, rewards, actn_log_probs = zip(*exp)
        actn_log_probs = torch.cat(actn_log_probs)
        state_vals = torch.cat(state_vals)

        if self.c_net_type == 'lstm':
            trmnl_state_val, self.c_hdn_st = self.critic.forward(
                state,
                self.c_hdn_st,
            )
        else:
            trmnl_state_val = self.critic(state)

        trmnl_state_val = trmnl_state_val.item()
        returns = self._get_returns(trmnl_state_val, rewards).detach()
        adv = returns - state_vals
        actn_log_probs = actn_log_probs
        actor_loss = (-1.0 * actn_log_probs * adv.detach()).mean()
        critic_loss = adv.pow(2).mean()
        net_loss = (actor_loss + critic_loss).mean()

        # disable computing gradients
        if self.p_net_type == 'lstm':
            self.p_hdn_st = tuple([each.data for each in self.p_hdn_st])
        if self.c_net_type == 'lstm':
            self.c_hdn_st = tuple([each.data for each in self.c_hdn_st])

        self.actor_optmz.zero_grad()
        self.critic_optmz.zero_grad()

        if self.p_net_type == 'lstm':
            actor_loss.backward(retain_graph=True)
        else:
            actor_loss.backward()

        critic_loss.backward()
        self.actor_optmz.step()
        self.critic_optmz.step()

        return net_loss.item(), ep_reward/ts, profit/ts, bal/ts, units_held/ts, net_worth/ts

    def evaluate(self, start_dt, duration, show_logs=False, show_pred=False):
        idx = self.env.df.loc[self.env.df['date'] == start_dt].index[0]
        rewards = deque(maxlen=duration)
        profits = deque(maxlen=duration)
        bals = deque(maxlen=duration)
        units_held_l = deque(maxlen=duration)
        losses = deque(maxlen=duration)
        net_worth_l = deque(maxlen=duration)
        buy_steps = []
        sell_steps = []
        buy_prices = []
        sell_prices = []
        actions = []
        want_to_buy_prices = []
        want_to_buy_steps = []
        want_to_sell_prices = []
        want_to_sell_steps = []
        state = self.env.reset(idx)
        state = FT(state)

        for _ in range(duration):
            if self.p_net_type == 'lstm':
                ip_state = self.actor.init_states(1)

            if self.p_net_type == 'lstm':
                policy, ip_state = self.actor.forward(
                    state,
                    ip_state,
                )
            else:
                policy = self.actor(state)

            actn, actn_log_prob = self._get_action(policy)
            nxt_state, reward, ep_ended, info = self.env.step(actn.item())
            ep_reward = info.get('reward')
            profit = info.get('profit')
            bal = info.get('balance')
            units_held = info.get('units_held')
            net_worth = info.get('net_worth')
            curr_step = info.get('curr_step')
            curr_price = info.get('curr_price')
            action = info.get('action')
            units_bought = info.get('units_bought')
            units_sold = info.get('units_sold')
            state = FT(nxt_state)

            if action == 0:
                if units_bought != 0:
                    buy_prices.append(curr_price)
                    buy_steps.append(curr_step)
                else:
                    want_to_buy_prices.append(curr_price)
                    want_to_buy_steps.append(curr_step)
            elif action == 1:
                if units_sold != 0:
                    sell_prices.append(curr_price)
                    sell_steps.append(curr_step)
                else:
                    want_to_sell_prices.append(curr_price)
                    want_to_sell_steps.append(curr_step)

            ep_reward = round(ep_reward, 2)
            profit = round(profit, 2)
            bal = round(bal, 2)
            net_worth = round(net_worth, 2)

            rewards.append(ep_reward)
            bals.append(bal)
            profits.append(profit)
            units_held_l.append(units_held)
            net_worth_l.append(net_worth)

        avg_net_worth = round(np.mean(net_worth_l), 2)
        avg_units_held = int(np.mean(units_held_l))
        avg_profit = round(np.mean(profits), 2)
        avg_bal = round(np.mean(bals), 2)
        avg_reward = round(np.mean(rewards), 2)
        avg_loss = round(np.mean(losses), 2)
        max_gains = round(max(profits), 2)

        self.eval_logs['reward'] = ep_reward
        self.eval_logs['r_avg_loss'] = avg_loss
        self.eval_logs['r_avg_net_worth'] = avg_net_worth
        self.eval_logs['r_avg_profit'] = avg_profit
        self.eval_logs['r_avg_bal'] = avg_bal
        self.eval_logs['r_avg_units_held'] = avg_units_held

        self.env.render(
            buy_steps,
            buy_prices,
            sell_steps,
            sell_prices,
            want_to_buy_prices,
            want_to_buy_steps,
            want_to_sell_prices,
            want_to_sell_steps,
            idx,
            self.env.curr_step,
            show_logs,
            show_pred,
        )

        print(f'Avg.Rewards: {avg_reward} | Max.Profit: {max_gains} | Avg.Profit: {avg_profit} | Avg.Units: {avg_units_held} ')
        return rewards, profits, actions

    def run(self, ep=1000):
        rewards = deque(maxlen=50)
        profits = deque(maxlen=50)
        bals = deque(maxlen=50)
        units_held_l = deque(maxlen=50)
        losses = deque(maxlen=50)
        net_worth_l = deque(maxlen=50)

        for ep_no in range(ep):
            ep_loss, ep_reward, profit, bal, units_held, net_worth = self.train()
            ep_loss = round(ep_loss, 3)
            ep_reward = round(ep_reward, 2)
            profit = round(profit, 2)
            bal = round(bal, 2)
            net_worth = round(net_worth, 2)

            losses.append(ep_loss)
            avg_loss = round(np.mean(losses), 2)

            rewards.append(ep_reward)
            avg_reward = round(np.mean(rewards), 2)

            bals.append(bal)
            avg_bal = round(np.mean(bals), 2)

            profits.append(profit)
            avg_profit = round(np.mean(profits), 2)

            units_held_l.append(units_held)
            avg_units_held = int(np.mean(units_held_l))

            net_worth_l.append(net_worth)
            avg_net_worth = round(np.mean(net_worth_l), 2)

            # save logs for analysis
            rewards.append(ep_reward)
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['r_avg_reward'] = avg_reward
            self.logs[ep_no]['r_avg_loss'] = avg_loss
            self.logs[ep_no]['r_avg_net_worth'] = avg_net_worth
            self.logs[ep_no]['r_avg_profit'] = avg_profit
            self.logs[ep_no]['r_avg_bal'] = avg_bal
            self.logs[ep_no]['r_avg_units_held'] = avg_units_held

            if ep_no % self.log_freq == 0:
                print(f'\nEp: {ep_no} | TS: {self.curr_step} | L: {ep_loss} | R: {ep_reward} | P: {profit} | R.Avg P: {avg_profit} | NW: {net_worth} | R.Avg NW: {avg_net_worth} | R.U: {avg_units_held}', end='')
                