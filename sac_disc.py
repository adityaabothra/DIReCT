import copy
from collections import defaultdict
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import FloatTensor as FT, tensor as T
from torch.optim import Adam

from networks.sac.discrete.policy_net import PolicyNetwork
from networks.sac.discrete.q_net import QNetwork
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
    ):
        self.env = env
        self.env_name = name
        self.input_dim = input_dim
        self.action_space = env.action_space
        self.hyprprms = hyprprms
        self.eps = self.hyprprms.get('eps', 1e-6)
        self.lr = self.hyprprms.get('lr', 5e-4)
        self.gamma = self.hyprprms.get('gamma', 0.99)
        self.eval_ep = self.hyprprms.get('eval_ep', 50)
        self.mem_sz = self.hyprprms.get('mem_sz', 20000)
        self.bth_sz = self.hyprprms.get('bth_sz', 2000)
        self.critic_sync_f = self.hyprprms.get('critic_sync_f', 5)
        self.tau = self.hyprprms.get('tau', 1e-2)
        self.save_mdls = save_mdls
        self.load_mdls = load_mdls
        self.memory = ReplayBuffer(self.mem_sz)
        self.tgt_entropy = -self.action_space.n

        # alpha network
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam(params=[self.log_alpha], lr=self.lr)

        # policy network
        self.policy = networks.get(
            'policy_net',
            PolicyNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
                eps=self.eps,
            ),
        )
        self.policy_optmz = optmzrs.get(
            'policy_optmz',
            Adam(self.policy.parameters(), lr=self.lr),
        )

        # critic network
        self.critic_a = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_a_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a.parameters(), lr=self.lr),
        )

        self.critic_b = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_b_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b.parameters(), lr=self.lr),
        )
        
        if self.load_mdls:
            self._load_models()

        # target critic network
        self.critic_a_tgt = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_a_tgt.load_state_dict(self.critic_a.state_dict())
        self.critic_a_tgt_optmz = optmzrs.get(
            'critic_a_optmz',
            Adam(self.critic_a_tgt.parameters(), lr=self.lr),
        )

        self.critic_b_tgt = networks.get(
            'q_net',
            QNetwork(
                state_dim=input_dim,
                action_dim=self.action_space.n,
            )
        )
        self.critic_b_tgt.load_state_dict(self.critic_b.state_dict())
        self.critic_b_tgt_optmz = optmzrs.get(
            'critic_b_optmz',
            Adam(self.critic_b_tgt.parameters(), lr=self.lr),
        )

        self.log_freq = log_freq
        self.logs = defaultdict(
            lambda: {
                'reward': 0,
                'cum_reward': 0,
            },
        )
        self.eval_logs = defaultdict(
            lambda: {
                'reward': 0,
                'cum_reward': 0,
            },
        )

    def _get_action(self, state):
        state = torch.cat([state])
        actions, _, _ = self.policy.sample(state)
        return actions

    def _sync_weights(self, src, tgt):
        for target_param, local_param in zip(tgt.parameters(), src.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def _save_models(
        self,
        policy_path=None,
        critic_b_path=None,
        critic_a_path=None,
        alpha_path=None,
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
        tgt_value_path=None,
        value_path=None,
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

    def _get_alpha_loss(self, log_probs):
        alpha_loss = - (self.log_alpha.exp() * (self.tgt_entropy + log_probs.detach())).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        return alpha_loss.cpu().detach().numpy()

    def _get_critic_loss(self, states, actions, rewards, nxt_states, dones):
        with torch.no_grad():
            _, probs, log_probs = self.policy.sample(nxt_states)
            q1_target = self.critic_a_tgt(nxt_states)
            q2_target = self.critic_b_tgt(nxt_states)
            min_q_vals = torch.min(q1_target, q2_target) 
            next_q = ((probs * min_q_vals) - self.alpha * log_probs).sum(1).unsqueeze(-1)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * next_q

        pred_q1_values = self.critic_a(states).gather(1, actions.unsqueeze(-1).long())
        pred_q2_values = self.critic_b(states).gather(1, actions.unsqueeze(-1).long())

        critic_a_loss = F.mse_loss(pred_q1_values, target_q)
        critic_b_loss = F.mse_loss(pred_q2_values, target_q)

        self.critic_a_optmz.zero_grad()
        self.critic_b_optmz.zero_grad()

        critic_a_loss.backward()
#         clip_grad_norm_(self.critic_a.parameters(), 1)
        critic_b_loss.backward()
#         clip_grad_norm_(self.critic_b.parameters(), 1)

        self.critic_a_optmz.step()
        self.critic_b_optmz.step()

        self._sync_weights(self.critic_a, self.critic_a_tgt)
        self._sync_weights(self.critic_b, self.critic_b_tgt)

    def _get_policy_loss(self, states, alpha):
        _, probs, log_probs = self.policy.sample(states)
        with torch.no_grad():
            pred_q1_values = self.critic_a(states)
            pred_q2_values = self.critic_b(states)
            min_q_values = torch.min(pred_q1_values, pred_q2_values)

        inside_term = alpha * log_probs - min_q_values
        policy_loss = (probs * inside_term).sum(dim=1).mean()
        log_action_pi = torch.sum(log_probs * probs, dim=1)

        self.policy_optmz.zero_grad()
        policy_loss.backward()
        self.policy_optmz.step()

        return policy_loss.cpu().detach().numpy(), log_action_pi

    def train(self, ep_no):
        states, actions, rewards, nxt_states, dones = \
            self.memory.sample(self.bth_sz)

        rewards = T(rewards, dtype=torch.float, device=DEVICE)
        dones = T(dones, dtype=torch.float, device=DEVICE)
        nxt_states = T(nxt_states, dtype=torch.float, device=DEVICE)
        states = T(states, dtype=torch.float, device=DEVICE)
        actions = T(actions, dtype=torch.float, device=DEVICE)

        curr_alpha = self.alpha.clone()
        policy_loss, log_probs = self._get_policy_loss(
            states,
            curr_alpha,
        )
        alpha_loss = self._get_alpha_loss(log_probs)
        self._get_critic_loss(
            states,
            actions,
            rewards,
            nxt_states,
            dones,
        )

        return alpha_loss, policy_loss
    
    def evaluate(self, ep=None):
        if not ep:
            ep = self.eval_ep

        for ep_no in range(ep):
            state = self.env.reset()
            state = T(state, device=DEVICE)
            ep_ended = False
            ep_reward = 0
            ts = 0

            while not ep_ended and ts < 200:
                action = self._get_action(state)
                nxt_state, reward, ep_ended, _ = self.env.step(action.item())
                nxt_state = T(nxt_state, device=DEVICE)
                ep_reward += reward
                state = nxt_state
                ts += 1

            self.eval_logs[ep_no]['reward'] = ep_reward

    def run(self, ep=1000):
        print('collecting experience...')
        rewards = []

        for ep_no in range(ep):
            state = self.env.reset()
            state = T(state, device=DEVICE)
            ep_ended = False
            ep_reward = 0
            a_loss, c_loss, p_loss = 0, 0, 0
            ts = 0

            while not ep_ended and ts < 200:
                action = self._get_action(state)
                nxt_state, reward, ep_ended, _ = self.env.step(action.item())
                state = T(state, device=DEVICE)
                ep_reward += reward
                nxt_state = T(nxt_state, device=DEVICE)
                reward = T(reward, device=DEVICE)
                ep_ended = T(ep_ended, device=DEVICE)
                self.memory.add((state, action, reward, nxt_state, ep_ended))
                state = nxt_state
                if self.memory.curr_size > self.mem_sz:
                    a_loss, p_loss = self.train(ep_no)

                    if ep_no % 100:
                        self._save_models()
                ts += 1

            rewards.append(ep_reward)
            avg_reward = np.mean(rewards[-50:])
            self.logs[ep_no]['reward'] = ep_reward
            self.logs[ep_no]['avg_reward'] = avg_reward

            if ep_no % self.log_freq == 0:
                if self.memory.curr_size > self.mem_sz:
                    print(f'Episode: {ep_no}, Reward: {ep_reward}, Avg. Reward: {avg_reward}, Policy Loss={round(float(p_loss), 2)}')
                else:
                    print(ep_no, end='..')





