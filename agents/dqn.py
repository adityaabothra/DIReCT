from collections import defaultdict, deque
import os

import numpy as np
import torch


class ReplayMemory(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.curr_size = 0

    def __len__(self):
        return len(self._storage)

    def store(self, exp):
        # exp = (state, action, reward, nxt_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(exp)
        else:
            self._storage[self._next_idx] = exp
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self.curr_size += 1

    def can_sample(self, size):
        return self.curr_size >= self._maxsize

    def _encode_sample(self, idxes):
        states, actions, rewards, nxt_states, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, action, reward, nxt_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nxt_states.append(nxt_state)
            dones.append(done)

        return np.array(states), np.array(actions), np.array(rewards), np.array(nxt_states), np.array(dones)

    def sample(self, batch_size):
        idxes = np.random.randint(0, len(self._storage) - 1, batch_size)
        return self._encode_sample(idxes)


class DQN:
    def __init__(
        self,
        env,
        target_net,
        policy_net,
        optimizer, 
        loss_func,
        model_path,
        n_actions,
        env_type='vector',
        log_freq=10,
        tau = 1e-3,
        train_freq=5,
        w_sync_freq=5,
        batch_size=10,
        roll_n=20,
        memory_size=5000,
        gamma=0.95,
        step_size=0.001,
        episodes=1000,
        eval_episodes=50,
        epsilon_start=0.8,
        epsilon_decay=0.9996,
        epsilon_min=0.01,
        load_pretrained=False,
        save_pretrained=False,
        network_type='nn',
    ):
        self.env = env
        self.env_type = env_type
        self.n_actions = n_actions
        self.actions = range(n_actions)
        self.roll_n = roll_n
        self.gamma = np.float64(gamma)
        self.model_path = model_path
        self.policy_net = policy_net
        self.target_net = target_net
        self.save_pretrained = save_pretrained
        if load_pretrained and os.path.exists(f'{model_path}/policy_net') and os.path.exists(f'{model_path}/target_net'):
            print('Pretrained Models loaded')
            self.policy_net.load_state_dict(torch.load(f'{model_path}/policy_net'))
            self.target_net.load_state_dict(torch.load(f'{model_path}/target_net'))

        self.memory_size = memory_size
        self.network_type = network_type
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(size=memory_size)
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.step_size = step_size
        self.tau = tau
        self.episodes = episodes
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eval_episodes = eval_episodes
        self.w_sync_freq = w_sync_freq
        self.train_freq = train_freq
        self.log_freq = log_freq
        self.batch_no = 0
        self.load_pretrained = load_pretrained
        self.curr_step = 0

        # initialize action-value function
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions),
        )

        # initialize traning logs
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'cumulative_reward': 0,
                'epsilon': None,
                'profit': 0,
                'balance': 0,
                'units_held': 0,
            },
        )

        # initialize evaluation logs
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'cumulative_reward': 0,
                'epsilon': None,
                'profit': 0,
                'balance': 0,
                'units_held': 0,
            },
        )

        if self.network_type == 'lstm':
            self.hdn_st = self.policy_net.init_states(1)
            self.tgt_hdn_st = self.target_net.init_states(1)

            self.train_hdn_st = self.policy_net.init_states(self.batch_size)
            self.train_tgt_hdn_st = self.target_net.init_states(self.batch_size)

    def _clip_reward(self, reward):
        return (2 * (
            reward - self.env.min_reward
        ) / (self.env.max_reward - self.env.min_reward)) - 1

    def _get_action_probs(self, state, epsilon):
        # initialize episilon probability to all the actions
        probs = np.ones(self.n_actions) * (epsilon / self.n_actions)
        if self.network_type == 'lstm':
            action_values, self.hdn_st = self.policy_net.forward(
                state.unsqueeze(0),
                self.hdn_st,
            )
        else:
            action_values = self.policy_net.forward(state.unsqueeze(0))
        best_action = torch.argmax(action_values)
        # initialize 1-epsilon probability to the greedy action
        probs[best_action] = 1 - epsilon + (epsilon / self.n_actions)
        return probs

    def _get_action(self, state, epsilon):
        if self.env_type == 'image':
            oned_state, state = state
        action = np.random.choice(
            self.actions, 
            p=self._get_action_probs(
                torch.from_numpy(state).float(),
                epsilon,
            ),
        )

        return action, self.actions.index(action)

    def _store_transition(self, transition):
        self.replay_memory.store(transition)

    def _train_one_batch(self, transitions, epsilon):
        states, actions, rewards, next_states, goal_achieved = transitions

        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        actions = torch.from_numpy(np.array([actions])).view(-1, 1)
        rewards = torch.from_numpy(np.array([rewards])).float().view(-1, 1)
        goal_achieved = torch.from_numpy(np.array([goal_achieved])).view(-1, 1).float()

        if self.network_type == 'lstm':
            Q_values, self.train_hdn_st = self.policy_net(
                states,
                self.train_hdn_st,
            )
        else:
            Q_values = self.policy_net(states)

        predictions = Q_values.gather(1, actions)
        if self.network_type == 'lstm':
            actions, self.train_tgt_hdn_st = self.target_net(
                next_states,
                self.train_tgt_hdn_st,
            )
            labels_next = torch.max(
                actions,
                dim=1,
            ).values.view(-1, 1).detach()
        else:
            labels_next = torch.max(
                self.target_net(next_states),
                dim=1,
            ).values.view(-1, 1).detach()

        if self.network_type == 'lstm':
            self.train_hdn_st = tuple([each.data for each in self.train_hdn_st])

        labels = rewards + (self.gamma * labels_next * (1 - goal_achieved))

        loss = self.loss_func(predictions, labels)
        self.optimizer.zero_grad()

        if self.network_type == 'lstm':
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    def _sync_weights(self, soft=False):
        if not soft:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # @TODO: Implement soft updates
            pass

    def run(self, ep=None):
        if not ep:
            ep = self.episodes

        epsilon = self.epsilon_start
        rewards = deque(maxlen=self.roll_n)
        profits = deque(maxlen=self.roll_n)
        bals = deque(maxlen=self.roll_n)
        units_held_l = deque(maxlen=self.roll_n)
        losses = deque(maxlen=self.roll_n)
        net_worth_l = deque(maxlen=self.roll_n)

        for ep_no in range(ep):
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            ep_ended = False
            ep_reward = 0
            ep_loss = 0
            timestep = 0
            profit = 0
            net_worth = 0
            bal = 0
            units_held = 0
            state = self.env.reset()

            while not ep_ended:
                action, action_idx = self._get_action(state, epsilon)
                next_state, reward, ep_ended, info = self.env.step(action=[action, 1])
                ep_reward += info.get('reward')
                profit += info.get('profit')
                bal += info.get('balance')
                units_held += info.get('units_held')
                net_worth += info.get('net_worth')

                self._store_transition(
                    [state, action_idx, reward, next_state, ep_ended]
                )

                if self.replay_memory.can_sample(self.batch_size):
                    transitions = self.replay_memory.sample(self.batch_size)
                    if timestep % self.train_freq == 0:
                        ep_loss += self._train_one_batch(transitions, epsilon)

                    if self.batch_no % self.w_sync_freq == 0:
                        self._sync_weights()
                    self.batch_no += 1

                timestep += 1
                state = next_state
                self.curr_step += 1

            if self.replay_memory.can_sample(self.batch_size):
                ep_reward = round(ep_reward/timestep, 2)
                ep_loss = round(ep_loss, 3)
                avg_p = round(profit/timestep, 2)
                avg_b = round(bal/timestep, 2)
                net_worth = round(net_worth/timestep, 2)
                avg_u_h = int(units_held/timestep)

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
                if self.replay_memory.can_sample(self.batch_size):
                    print(f'\nEp: {ep_no} | TS: {self.curr_step} | L: {ep_loss} | R: {ep_reward} | P: {avg_p} | R.Avg P: {avg_profit} | NW: {net_worth} | R.Avg NW: {avg_net_worth} | R.U: {avg_units_held}', end='')
                else:
                    print(ep_no, end='..')

        if self.save_pretrained:
            self.save_models()

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

        for _ in range(duration):
            action, action_idx = self._get_action(state, 0)
            nxt_state, reward, ep_ended, info = self.env.step([action, 1])
            ep_reward = info.get('reward')
            profit = info.get('profit')
            bal = info.get('balance')
            units_held = info.get('units_held')
            net_worth = info.get('net_worth')
            curr_step = info.get('curr_step')
            curr_price = info.get('curr_price')
            units_bought = info.get('units_bought')
            units_sold = info.get('units_sold')

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

        self.eval_logs['reward'] = ep_reward
        self.eval_logs['r_avg_loss'] = avg_loss
        self.eval_logs['r_avg_net_worth'] = avg_net_worth
        self.eval_logs['r_avg_profit'] = avg_profit
        self.eval_logs['r_avg_bal'] = avg_bal
        self.eval_logs['r_avg_units_held'] = avg_units_held

        self.env.render(buy_steps, buy_prices, sell_steps, sell_prices, idx, self.env.curr_step, show_logs)

        print(f'Avg.Rewards: {avg_reward} | Avg.Profit: {avg_profit} | Avg.Units: {avg_units_held} | Avg.Bal: {avg_bal}')

    def save_models(self):
        torch.save(self.target_net.state_dict(), f'{self.model_path}/target_net')
        torch.save(self.policy_net.state_dict(), f'{self.model_path}/policy_net')