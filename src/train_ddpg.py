# src/train_ddpg.py
import os
import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_model import DDPGAgent
from src.environment import AgriEnv
from src.config import Config

def train_agent():
    # Initialize config and environment
    config = Config()
    env = AgriEnv(config)
    
    # Initialize agent with environment
    agent = DDPGAgent(env, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, noise_scale=0.4)
    
    # Train the agent
    total_timesteps = 20000  # Adjust as needed (e.g., increase to 50000 for better learning)
    batch_size = 64
    agent.train(total_timesteps=total_timesteps, batch_size=batch_size)
    
    # Save the trained model
    model_path = os.path.join(PROJECT_ROOT, "data", "rl_model.pth")
    agent.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_agent()