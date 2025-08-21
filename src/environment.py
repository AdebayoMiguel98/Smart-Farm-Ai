import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
from src.config import Config
from types import SimpleNamespace

class AgriEnv(Env):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_crops = len(config.crops)
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0] + [0.0] * self.num_crops),
            high=np.array([100.0, 3000.0, 1e6] + [100.0] * self.num_crops),
            shape=(3 + self.num_crops,),
            dtype=np.float32
        )
        self.action_space = Box(
            low=np.zeros(self.num_crops),
            high=np.ones(self.num_crops),
            dtype=np.float32
        )
        
        # Set environment spec
        self.spec = SimpleNamespace(max_episode_steps=config.params["simulation"]["years"])
        
        # Initialize state
        self.year = 0
        self.savings = config.params["simulation"]["initial_savings"]
        self.total_land = config.params["simulation"]["total_land"]
        self.temperature = config.params["environment"]["temperature"]
        self.rainfall = config.params["environment"]["rainfall"]
        self.weather_noise = 0.1
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.year = 0
        self.savings = self.config.params["simulation"]["initial_savings"]
        self.temperature = self.config.params["environment"]["temperature"]
        self.rainfall = self.config.params["environment"]["rainfall"]
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def _get_obs(self):
        allocations = np.zeros(self.num_crops)
        obs = np.concatenate([
            [self.year / self.config.params["simulation"]["years"],
             self.temperature / 40.0,
             self.savings / 1e6],
            allocations
        ]).astype(np.float32)
        if np.any(np.isnan(obs)):
            raise ValueError(f"NaN detected in observation: {obs}")
        return obs
    
    def _get_info(self):
        return {
            "allocations": np.zeros(self.num_crops),
            "crop_profits": {crop: 0.0 for crop in self.config.crops}
        }
    
    def step(self, action):
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum < 1e-6:  # Prevent division by near-zero
            action = np.ones_like(action) / len(action)  # Fallback to equal allocation
        else:
            action = action / (action_sum + 1e-6)  # Increased epsilon
        allocations = action * self.total_land
        
        # Simulate weather variations
        temp_variation = np.random.normal(0, self.weather_noise * self.temperature)
        rain_variation = np.random.normal(0, self.weather_noise * self.rainfall)
        current_temp = self.temperature + temp_variation
        current_rain = self.rainfall + rain_variation
        
        # Calculate yields and profits
        total_profit = 0.0
        crop_profits = {}
        for i, crop in enumerate(self.config.crops):
            crop_params = self.config.params["crops"][crop]
            base_yield = crop_params["yield"]["base"]
            temp_optimal = crop_params["yield"]["temp_optimal"]
            rain_optimal = crop_params["yield"]["rain_optimal"]
            
            temp_factor = 1.0 - 0.1 * abs(current_temp - temp_optimal) / temp_optimal
            rain_factor = 1.0 - 0.1 * abs(current_rain - rain_optimal) / rain_optimal
            crop_yield = max(0.0, base_yield * temp_factor * rain_factor)  # Ensure non-negative yield
            
            price = crop_params["market"]["price"]
            price_variation = np.random.normal(0, crop_params["market"]["volatility"] * price)
            current_price = max(0.5 * price, price + price_variation)  # Increased minimum price
            
            costs = sum(crop_params["cost"].values())
            revenue = crop_yield * current_price * allocations[i]
            profit = revenue - costs * allocations[i]
            crop_profits[crop] = profit
            total_profit += profit
        
        # Update savings
        if np.isnan(total_profit):
            raise ValueError(f"NaN detected in total_profit: {crop_profits}")
        self.savings = max(0, self.savings + total_profit)
        self.year += 1
        
        # Observation, reward, termination
        obs = np.concatenate([
            [self.year / self.config.params["simulation"]["years"],
             current_temp / 40.0,
             self.savings / 1e6],
            allocations / self.total_land
        ]).astype(np.float32)
        
        if np.any(np.isnan(obs)):
            raise ValueError(f"NaN detected in observation: {obs}")
        
        reward = total_profit
        if np.isnan(reward):
            raise ValueError(f"NaN detected in reward: {total_profit}")
        
        terminated = self.savings <= 0
        truncated = self.year >= self.config.params["simulation"]["years"]
        info = {
            "allocations": allocations,
            "crop_profits": crop_profits
        }
        
        return obs, reward, terminated, truncated, info
    
    def get_crop_params(self, crop: str):
        return self.config.params["crops"].get(crop, {})