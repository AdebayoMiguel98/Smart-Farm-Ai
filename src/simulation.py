import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from typing import Dict, List, Any
from src.rl_model import RLModel
from src.environment import AgriEnv
from src.config import Config

class Simulation:
    def __init__(self, env: AgriEnv, config: Config):
        self.env = env
        self.config = config
        self.model = RLModel(env)
        # Validate config.years
        if not hasattr(self.config, 'years') or not isinstance(self.config.years, int) or self.config.years <= 0:
            raise ValueError(f"Invalid 'years' in config: {getattr(self.config, 'years', 'missing')}. Expected positive integer.")

    def run_single(self) -> Dict[str, Any]:
        try:
            obs, _ = self.env.reset()
            results = {
                "years": [],
                "allocations": [],
                "profits": [],
                "savings": [],
                "crop_profits": {crop: [] for crop in self.config.crops}
            }
            
            for year in range(self.config.years):
                action = self.model.predict(obs, add_noise=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                results["years"].append(year + 1)
                results["allocations"].append(info["allocations"].copy())
                results["profits"].append(reward)
                results["savings"].append(self.env.savings)
                for crop, profit in info["crop_profits"].items():
                    results["crop_profits"][crop].append(profit)
                
                alloc_str = ", ".join([f"{crop}: {alloc:.2f}" for crop, alloc in zip(self.config.crops, info["allocations"])])
                profit_str = ", ".join([f"{crop}: {profit:.2f}" for crop, profit in info["crop_profits"].items()])
                print(f"Year {year+1}, Allocation: {{{alloc_str}}}, Profit: {reward:.2f}, Savings: {self.env.savings:.2f}, Crop Profits: {{{profit_str}}}")
                
                if terminated or truncated:
                    break
            
            return results
        except Exception as e:
            raise RuntimeError(f"Episode failed: {str(e)}")

    def run_multiple(self, num_episodes: int = 100) -> Dict[str, List[Any]]:
        try:
            if not os.path.exists("data/rl_model.pth"):
                print("Training RL model...")
                self.model.train(total_timesteps=20000)
                self.model.save()
            else:
                self.model.load()
            
            results = {
                "years": list(range(1, self.config.years + 1)),
                "allocations": [],
                "profits": [],
                "savings": [],
                "crop_profits": []
            }
            
            for _ in tqdm(range(num_episodes), desc="Running simulations"):
                episode = self.run_single()
                results["allocations"].append(episode["allocations"])
                results["profits"].append(episode["profits"])
                results["savings"].append(episode["savings"])
                results["crop_profits"].append(episode["crop_profits"])
            
            # Average results across episodes
            avg_results = {
                "years": results["years"],
                "allocations": np.mean(results["allocations"], axis=0).tolist(),
                "profits": np.mean(results["profits"], axis=0).tolist(),
                "savings": np.mean(results["savings"], axis=0).tolist(),
                "crop_profits": {crop: np.mean([ep[crop] for ep in results["crop_profits"]], axis=0).tolist()
                                 for crop in self.config.crops}
            }
            
            self.save_results(avg_results, "rl_results")
            return avg_results
        except Exception as e:
            raise RuntimeError(f"Multiple runs failed: {str(e)}")
    
    def run_baseline(self, num_episodes: int = 100) -> Dict[str, List[Any]]:
        try:
            results = {
                "years": list(range(1, self.config.years + 1)),
                "allocations": [],
                "profits": [],
                "savings": [],
                "crop_profits": []
            }
            
            for _ in tqdm(range(num_episodes), desc="Running baseline"):
                self.env.reset()
                equal_alloc = np.array([1.0 / self.config.num_crops] * self.config.num_crops)  # Equal ratios
                episode = {
                    "years": list(range(1, self.config.years + 1)),
                    "allocations": [],
                    "profits": [],
                    "savings": [],
                    "crop_profits": {crop: [] for crop in self.config.crops}
                }
                
                for year in range(self.config.years):
                    _, reward, terminated, truncated, info = self.env.step(equal_alloc)
                    episode["allocations"].append(info["allocations"].copy())
                    episode["profits"].append(reward)
                    episode["savings"].append(self.env.savings)
                    for crop, profit in info["crop_profits"].items():
                        episode["crop_profits"][crop].append(profit)
                    
                    alloc_str = ", ".join([f"{crop}: {alloc:.2f}" for crop, alloc in zip(self.config.crops, info["allocations"])])
                    profit_str = ", ".join([f"{crop}: {profit:.2f}" for crop, profit in info["crop_profits"].items()])
                    print(f"Baseline Year {year+1}, Allocation: {{{alloc_str}}}, Profit: {reward:.2f}, Savings: {self.env.savings:.2f}, Crop Profits: {{{profit_str}}}")
                    if terminated or truncated:
                        break
                
                # Pad if early termination
                current_length = len(episode["profits"])
                if current_length < self.config.years:
                    pad_length = self.config.years - current_length
                    episode["allocations"].extend([episode["allocations"][-1].copy() for _ in range(pad_length)])
                    episode["profits"].extend([0.0 for _ in range(pad_length)])
                    episode["savings"].extend([episode["savings"][-1] for _ in range(pad_length)])
                    for crop in self.config.crops:
                        episode["crop_profits"][crop].extend([0.0 for _ in range(pad_length)])
                
                results["allocations"].append(episode["allocations"])
                results["profits"].append(episode["profits"])
                results["savings"].append(episode["savings"])
                results["crop_profits"].append(episode["crop_profits"])
            
            # Average results across episodes
            avg_results = {
                "years": results["years"],
                "allocations": np.mean(results["allocations"], axis=0).tolist(),
                "profits": np.mean(results["profits"], axis=0).tolist(),
                "savings": np.mean(results["savings"], axis=0).tolist(),
                "crop_profits": {crop: np.mean([ep[crop] for ep in results["crop_profits"]], axis=0).tolist()
                                 for crop in self.config.crops}
            }
            
            self.save_results(avg_results, "baseline_results")
            return avg_results
        except Exception as e:
            raise RuntimeError(f"Baseline failed: {str(e)}")
    
    def get_crop_params(self, crop: str) -> Dict[str, Any]:
        return self.env.get_crop_params(crop)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        try:
            os.makedirs("data/outputs", exist_ok=True)
            # Flatten results for DataFrame
            flat_results = {
                "Year": results["years"],
                "Savings": results["savings"],
                "Profit": results["profits"]
            }
            for i, c in enumerate(self.config.crops):
                flat_results[f"{c}_Alloc"] = [a[i] for a in results["allocations"]]
            for c in self.config.crops:
                flat_results[f"{c}_Profit"] = results["crop_profits"][c]
            
            # Validate array lengths
            lengths = {k: len(v) for k, v in flat_results.items() if isinstance(v, list)}
            max_length = max(lengths.values())
            min_length = min(lengths.values())
            if max_length != min_length:
                print(f"Warning: Array lengths differ (min: {min_length}, max: {max_length}). Padding with NaN.")
                for k in flat_results:
                    if isinstance(flat_results[k], list) and len(flat_results[k]) < max_length:
                        flat_results[k] += [np.nan] * (max_length - len(flat_results[k]))
            
            df = pd.DataFrame(flat_results)
            df.to_csv(f"data/outputs/{filename}.csv", index=False)
            
            # Save config
            config_path = os.path.join("data", "current_config.json")
            if not hasattr(self.config, 'params') or not isinstance(self.config.params, dict):
                raise ValueError("Invalid config.params: Expected a dictionary")
            with open(config_path, "w") as f:
                json.dump(self.config.params, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Save failed: {str(e)}")