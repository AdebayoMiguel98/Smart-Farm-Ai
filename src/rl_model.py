import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from gymnasium.spaces import Box
from src.environment import AgriEnv
import random

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=-1)  # Output probabilities summing to 1
        return x

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DDPGAgent:
    def __init__(self, env: AgriEnv, actor_lr: float = 1e-4, critic_lr: float = 1e-3, 
                 gamma: float = 0.99, tau: float = 0.005, noise_scale: float = 0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale  # Set to 0.4 for more exploration
    
    def predict(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)
        if torch.any(torch.isnan(state)):
            raise ValueError(f"NaN detected in input state: {state}")
        action = self.actor(state).cpu().detach().numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)  # Clip to [0, 1]
            action_sum = np.sum(action)
            if action_sum < 1e-6:  # Prevent division by near-zero
                action = np.ones_like(action) / len(action)
            else:
                action = action / action_sum  # Normalize to sum to 1
        if np.any(np.isnan(action)):
            raise ValueError(f"NaN detected in action: {action}")
        return action
    
    def update(self, batch: tuple) -> None:
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(-1)
        
        if torch.any(torch.isnan(states)) or torch.any(torch.isnan(actions)) or torch.any(torch.isnan(rewards)):
            raise ValueError("NaN detected in batch: states, actions, or rewards")
        
        # Critic update
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        if torch.isnan(critic_loss):
            raise ValueError(f"NaN detected in critic_loss: {critic_loss}")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actor update
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        if torch.isnan(actor_loss):
            raise ValueError(f"NaN detected in actor_loss: {actor_loss}")
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str = "data/rl_model.pth") -> None:
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def load(self, path: str = "data/rl_model.pth") -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint and 'critic_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic_target.load_state_dict(self.critic.state_dict())
                print(f"Model loaded from {path} (standard format)")
            elif isinstance(checkpoint, collections.OrderedDict):
                self.actor.load_state_dict(checkpoint)
                self.actor_target.load_state_dict(checkpoint)
                state_dim = self.env.observation_space.shape[0]
                action_dim = self.env.action_space.shape[0]
                self.critic = Critic(state_dim, action_dim).to(self.device)
                self.critic_target = Critic(state_dim, action_dim).to(self.device)
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
                print(f"Model loaded from {path} (OrderedDict fallback: loaded as actor state dict, reinitialized critic)")
            else:
                raise ValueError(f"Unrecognized model checkpoint format: {type(checkpoint)}. Expected dict with 'actor_state_dict' and 'critic_state_dict' or OrderedDict for actor.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")
    
    def train(self, total_timesteps: int = 20000, batch_size: int = 64) -> None:
        try:
            replay_buffer = collections.deque(maxlen=100000)
            rewards = []
            episode_reward = 0
            episode_timesteps = 0
            episode_num = 0
            
            state, _ = self.env.reset()
            max_episode_steps = getattr(self.env.spec, 'max_episode_steps', self.env.config.params["simulation"]["years"])
            
            for t in range(total_timesteps):
                episode_timesteps += 1
                action = self.predict(state, add_noise=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                replay_buffer.append((state, action, [reward], next_state, [float(terminated or truncated)]))
                state = next_state
                episode_reward += reward
                
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards_batch, next_states, dones = zip(*batch)
                    batch = (np.array(states), np.array(actions), np.array(rewards_batch), np.array(next_states), np.array(dones))
                    self.update(batch)
                
                if terminated or truncated or episode_timesteps >= max_episode_steps:
                    rewards.append(episode_reward)
                    episode_num += 1
                    avg_reward = np.mean(rewards[-100:]) if rewards else np.nan
                    print(f"Episode {episode_num}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_timesteps = 0
            
            self.save()
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

  # Moved to top for proper scoping