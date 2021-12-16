from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import torch
from torch import tensor as T
from torch.optim import Adam

from networks.sac.continuous.policy_net import PolicyNetwork
from networks.sac.continuous.q_net import QNetwork
from helpers.replay_buffer import ReplayBuffer


DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


class SAC:
    def __init__(
        self,
        env,
        name,
        input_dim,
        networks={},
        optmzrs={},
        log_freq=10,
        hyprprms={},
        save_mdls=True,
        load_mdls=False,
        p_net_type='nn',
    ):
        self.env = env
        self.env_name = name
        self.p_net_type = p_net_type
        self.input_dim = input_dim
        self.action_space = env.action_space
        self.hyprprms = hyprprms
        self.eps = self.hyprprms.get('eps', 1e-6)
        self.lr = self.hyprprms.get('lr', 0.0004)
        self.gamma = self.hyprprms.get('gamma', 0.95)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.mem_sz = self.hyprprms.get('mem_sz', 5000)
        self.steps = self.hyprprms.get('steps', 5000)
        self.critic_sync_f = self.hyprprms.get('critic_sync_f', 5)
        self.tau = self.hyprprms.get('tau', 0.005)
        self.alpha = self.hyprprms.get('alpha', 0.2)
        self.batch_size = self.hyprprms.get('batch_size', self.mem_sz//10)
        self.save_mdls = save_mdls
        self.load_mdls = load_mdls
        self.memory = ReplayBuffer(self.mem_sz)
        self.curr_step = 0

        # policy network
        self.policy = networks.get(
            'policy_net',
            PolicyNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
                eps=self.eps,
            ),
        )
        self.policy.to(DEVICE)
        self.policy_optmz = optmzrs.get(
            'policy_optmz',
            Adam(self.policy.parameters(), lr=self.lr),
        )

        # critic network
        self.critic_a = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.critic_a.to(DEVICE)
        self.critic_a_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a.parameters(), lr=self.lr),
        )

        self.critic_b = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.critic_b.to(DEVICE)
        self.critic_b_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b.parameters(), lr=self.lr),
        )

        # target critic network
        self.tgt_critic_a = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.tgt_critic_a.to(DEVICE)

        self.tgt_critic_b = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.shape[0],
            )
        )
        self.tgt_critic_b.to(DEVICE)

        self.log_freq = log_freq
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'avg_reward': 0,
            },
        )

        if self.p_net_type == 'lstm':
            self.p_hdn_st = self.policy.init_states(1)

    def _get_action(self, state, add_noise=True):
        state = torch.cat([state]).float()
        if self.p_net_type == 'lstm':
            actions, _, self.p_hdn_st = self.policy.sample(state, self.p_hdn_st, add_noise)
        else:
            actions, _ = self.policy.sample(state, add_noise)

        return actions.cpu().detach().numpy()

    def _sync_weights(self, src, tgt):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

    def _get_q(self, states, actions):
        q1 = self.critic_a(states, actions)
        q2 = self.critic_b(states, actions)
        return q1, q2

    def _get_target_q(self, states, actions, rewards, nxt_states, dones):
        with torch.no_grad():
            if self.p_net_type == 'lstm':
                nxt_actions, log_probs, self.p_hdn_st = self.policy.sample(nxt_states, self.p_hdn_st)
            else:
                nxt_actions, log_probs = self.policy.sample(nxt_states)
            nxt_q1 = self.tgt_critic_a(nxt_states, nxt_actions)
            nxt_q2 = self.tgt_critic_b(nxt_states, nxt_actions)
            nxt_q = torch.min(nxt_q1, nxt_q2) - self.alpha * log_probs

        target_q = rewards + (1.0 - dones) * self.gamma * nxt_q

        return target_q

    def _save_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
    ):
        path = Path(f'../models/{self.env_name}/')
        if not path.exists():
            path.mkdir()

        if policy_path is None:
            policy_path = path/'actor'

        if critic_a_path is None:
            critic_a_path = path/'critic_a'

        if critic_b_path is None:
            critic_b_path = path/'critic_b'

        self.policy.save(policy_path)
        self.critic_a.save(critic_a_path)
        self.critic_b.save(critic_b_path)

    def _load_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
    ):
        print('loading models....')
        path = Path(f'../models/{self.env_name}/')

        if policy_path is not None:
            policy_path = path/'actor'
            self.policy.load(policy_path)

        if critic_a_path is not None:
            critic_a_path = path/'critic_a'
            self.critic_a.load(critic_a_path)

        if critic_b_path is not None:
            critic_b_path = path/'critic_b'
            self.critic_b.load(critic_b_path)

    def _train_critic_net(self, states, actions, rewards, nxt_states, dones):
        pred_q1, pred_q2 = self._get_q(states, actions)
        target_q = self._get_target_q(states, actions, rewards, nxt_states, dones)

        critic_a_loss = torch.mean((pred_q1 - target_q).pow(2))
        critic_b_loss = torch.mean((pred_q2 - target_q).pow(2))

        self.critic_a_optmz.zero_grad()
        critic_a_loss.backward()
        self.critic_a_optmz.step()

        self.critic_b_optmz.zero_grad()
        critic_b_loss.backward()
        self.critic_b_optmz.step()

    def _train_policy_net(self, states):
        if self.p_net_type == 'lstm':
            actions, entropy, self.p_hdn_st = self.policy.sample(states, self.p_hdn_st)
        else:
            actions, entropy = self.policy.sample(states)
        entropy = entropy.view(-1)

        pred_q1 = self.critic_a.forward(states, actions)
        pred_q2 = self.critic_b.forward(states, actions)
        q = torch.min(pred_q1, pred_q2).view(-1)
        policy_loss = torch.mean(self.alpha * entropy - q)

        self.policy_optmz.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optmz.step()

        return policy_loss.cpu().detach().numpy()

    def train(self, ep_no):
        states, actions, rewards, nxt_states, dones = \
            self.memory.sample(self.batch_size)

        rewards = rewards.float().to(DEVICE)
        dones = dones.float().to(DEVICE)
        nxt_states = nxt_states.float().to(DEVICE)
        states = states.float().to(DEVICE)
        actions = actions.float().to(DEVICE)

        self._train_critic_net(
            states,
            actions,
            rewards,
            nxt_states,
            dones,
        )
        policy_loss = self._train_policy_net(states)

        if ep_no % self.critic_sync_f:
            self._sync_weights(self.critic_a, self.tgt_critic_a)
            self._sync_weights(self.critic_b, self.tgt_critic_b)

        return policy_loss

    def evaluate(self, start_dt, duration, show_logs=False):
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
        state = self.env.reset(idx)
        state = T(state, dtype=torch.float, device=DEVICE)

        for _ in range(duration):
            action = self._get_action(state)
            if self.p_net_type == 'lstm':
                action = action[0]
            nxt_state, reward, ep_ended, info = self.env.step(action)
            ep_reward = info.get('reward')
            profit = info.get('profit')
            bal = info.get('balance')
            units_held = info.get('units_held')
            net_worth = info.get('net_worth')
            curr_step = info.get('curr_step')
            curr_price = info.get('curr_price')
            units_bought = info.get('units_bought')
            units_sold = info.get('units_sold')
            state = T(nxt_state, device=DEVICE)

            if units_bought != 0:
                buy_prices.append(curr_price)
                buy_steps.append(curr_step)
            elif units_sold != 0:
                sell_prices.append(curr_price)
                sell_steps.append(curr_step)

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
        net_gains = round(sum(profits), 2)

        self.eval_logs['reward'] = ep_reward
        self.eval_logs['r_avg_loss'] = avg_loss
        self.eval_logs['r_avg_net_worth'] = avg_net_worth
        self.eval_logs['r_avg_profit'] = avg_profit
        self.eval_logs['r_avg_bal'] = avg_bal
        self.eval_logs['r_avg_units_held'] = avg_units_held

        self.env.render(buy_steps, buy_prices, sell_steps, sell_prices, idx, self.env.curr_step, show_logs)

        print(f'Avg.Rewards: {avg_reward} | Tot.Profit: {net_gains} | Avg.Profit: {avg_profit} | Avg.Units: {avg_units_held} ')

    def run(self, ep=1000):
        rewards = deque(maxlen=50)
        profits = deque(maxlen=50)
        bals = deque(maxlen=50)
        units_held_l = deque(maxlen=50)
        losses = deque(maxlen=50)
        net_worth_l = deque(maxlen=50)

        if self.load_mdls:
            self._load_models(
                policy_path=f'models/{env_name}/policy',
                critic_a_path=f'models/{env_name}/critic_a',
                critic_b_path=f'models/{env_name}/critic_b',
                value_path=f'models/{env_name}/value',
                tgt_value_path=f'models/{env_name}/tgt_value',
            )

        for ep_no in range(ep):
            state = self.env.reset()
            state = T(state, device=DEVICE)
            ep_ended = False
            ep_reward = 0
            ep_loss = 0
            ts = 0
            net_worth = 0
            profit = 0
            bal = 0
            units_held = 0

            while not ep_ended:
                action = self._get_action(state)
                if self.p_net_type == 'lstm':
                    action = action[0]
                nxt_state, reward, ep_ended, info = self.env.step(action)

                ep_reward += reward
                profit += info.get('profit')
                bal += info.get('balance')
                units_held += info.get('units_held')
                net_worth += info.get('net_worth')

                action = T(action, device=DEVICE)
                reward = T(reward, device=DEVICE)
                nxt_state = T(nxt_state, device=DEVICE)
                ep_ended = T(ep_ended, device=DEVICE)

                self.memory.add((state, action, reward, nxt_state, ep_ended))
                state = nxt_state

                if self.memory.curr_size > self.mem_sz:
                    ep_loss += self.train(ep_no)
                    if ep_no % 100:
                        self._save_models()
                ts += 1
                self.curr_step += 1

            if self.memory.curr_size > self.mem_sz:
                ep_reward = round(ep_reward/ts, 2)
                ep_loss = round(ep_loss, 2)
                avg_p = round(profit/ts, 2)
                avg_b = round(bal/ts, 2)
                avg_u_h = int(units_held/ts)

                losses.append(ep_loss)
                avg_loss = round(np.mean(losses), 2)

                rewards.append(ep_reward)
                avg_reward = round(np.mean(rewards), 2)

                bals.append(avg_b)
                avg_bal = round(np.mean(bals), 2)

                profits.append(avg_p)
                avg_profit = round(np.mean(profits), 2)

                units_held_l.append(avg_u_h)
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

            if ep_no == 0:
                print('collecting experience...')
            if ep_no % self.log_freq == 0:
                if self.memory.curr_size > self.mem_sz:
                    print(f'\nEp: {ep_no} | TS: {self.curr_step} | L: {ep_loss} | R: {ep_reward} | R.Avg.R: {avg_reward} | P: {avg_p} | R.Avg P: {avg_profit} | B: {avg_b} | R.Avg B: {avg_bal} | R.Avg.U: {avg_units_held}', end='')
                else:
                    print(ep_no, end='..')




