import os
import torch
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
from src.config import Config
from src.environment import AgriEnv
from src.rl_model import Actor, Critic

class DDPG:
    def __init__(self, env: Env, config: Config, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, noise_std=0.2):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(len(config.crops)).to(self.device)
        self.critic = Critic(len(config.crops)).to(self.device)
        self.actor_target = Actor(len(config.crops)).to(self.device)
        self.critic_target = Critic(len(config.crops)).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.action_dim = len(config.crops)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        noise = np.random.normal(0, self.noise_std, size=self.action_dim)
        action = action + noise
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

def train_ddpg(config, episodes=1000, save_path="data/rl_model.pth"):
    env = AgriEnv(config)
    agent = DDPG(env, config)
    replay_buffer = ReplayBuffer()
    total_rewards = []
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            agent.update(replay_buffer)
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    # Save the actor model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Replace current save code with this:
        torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
             }, save_path)
        print(f"Model saved to {save_path} (state_dicts for actor & critic)")




if __name__ == "__main__":
    config = Config()
    train_ddpg(config)