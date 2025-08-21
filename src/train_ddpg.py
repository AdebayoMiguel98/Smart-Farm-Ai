import numpy as np
import matplotlib.pyplot as plt
from src.environment import AgriEnv
from src.config import Config
from src.rl_model import RLModel

def train_agent(total_timesteps: int = 20000, batch_size: int = 64, noise_scale: float = 0.3):
    try:
        config = Config()
        env = AgriEnv(config)
        model = RLModel(env, noise_scale=noise_scale)
        
        model.train(total_timesteps=total_timesteps, batch_size=batch_size)
        
        rewards = []
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        state, _ = env.reset()
        
        for t in range(total_timesteps):
            episode_timesteps += 1
            action = model.predict(state, add_noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated or episode_timesteps >= env.spec.max_episode_steps:
                rewards.append(episode_reward)
                episode_num += 1
                alloc_str = ", ".join([f"{crop}: {alloc:.2f}" for crop, alloc in zip(config.crops, info["allocations"])])
                print(f"Episode {episode_num}, Reward: {episode_reward:.2f}, Avg Reward: {np.mean(rewards[-100:]):.2f}, "
                      f"Final Savings: {env.savings:.2f}, Allocation: {{{alloc_str}}}")
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
        
        # Plot training rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards Over Time")
        plt.grid(True)
        plt.savefig("data/training_rewards.png")
        plt.close()
        
        # Save model
        model.save("data/rl_model.pth")
        print("Training completed and model saved to data/rl_model.pth")
        
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")

if __name__ == "__main__":
    train_agent()