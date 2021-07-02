import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from env import Environments
from model import PPO
from process import test
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil


LOGS = "Logs/"
MODELS = "Models/"

def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    mp = _mp.get_context("spawn")
    envs = Environments(1, 1, 12)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()

    process = mp.Process(target=test, args=(model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    ep = 0
    while True:
        ep += 1
        policy_history, actions, values, states, rewards, actions_taken = [], [], [], [], [], []
        for _ in range(512):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            policy_history.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, taken, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                taken = torch.cuda.FloatTensor(taken)
            else:
                reward = torch.FloatTensor(reward)
                taken = torch.FloatTensor(taken)
            rewards.append(reward)
            actions_taken.append(taken)
            curr_states = state

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        policy_history = torch.cat(policy_history).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        q_value = 0
        positive_rewards = []
        for value, reward, taken in list(zip(values, rewards, actions_taken))[::-1]:
            q_value = q_value * 0.99 * 1.0
            q_value = q_value + reward + 0.99 * next_value.detach() * (1 - taken) - value.detach()
            next_value = value
            positive_rewards.append(q_value + value)
        positive_rewards = positive_rewards[::-1]
        positive_rewards = torch.cat(positive_rewards).detach()
        advantages = positive_rewards - values
        for i in range(10):
            indice = torch.randperm(512 * 12)
            for j in range(8):
                batch_indices = indice[
                                int(j * (512 * 12 / 8)): int((j + 1) * (
                                        512 * 12 / 8))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - policy_history[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) *
                                                   advantages[
                                                       batch_indices]))
                critic_loss = F.smooth_l1_loss(positive_rewards[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - 0.01 * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        print("Current Episode: {}. Total loss: {}".format(ep, total_loss))


if __name__ == "__main__":
    train()
