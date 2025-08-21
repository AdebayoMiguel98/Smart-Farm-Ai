import json
import os
from copy import deepcopy

class Config:
    def __init__(self, config_file="data/default_params.json"):
        # Default parameters with Tomato instead of Soybeans
        self.params = {
            "simulation": {
                "years": 20,
                "initial_savings": 10000.0,
                "total_land": 100.0
            },
            "environment": {
                "temperature": 25.0,
                "rainfall": 1000.0
            },
            "crops": {
                "maize": {
                    "cost": {"seed": 35.0, "fertilizer": 60.0, "pesticide": 25.0, "labor": 90.0},  # Total: 210
                    "market": {"price": 450.0, "volatility": 0.15},
                    "yield": {"base": 6.0, "temp_optimal": 25.0, "rain_optimal": 800.0}
                },
                "rice": {
                    "cost": {"seed": 40.0, "fertilizer": 70.0, "pesticide": 30.0, "labor": 110.0},  # Total: 250
                    "market": {"price": 600.0, "volatility": 0.2},
                    "yield": {"base": 5.0, "temp_optimal": 27.0, "rain_optimal": 1200.0}
                },
                "tomato": {
                    "cost": {"seed": 50.0, "fertilizer": 65.0, "pesticide": 35.0, "labor": 120.0},  # Total: 270
                    "market": {"price": 800.0, "volatility": 0.25},
                    "yield": {"base": 20.0, "temp_optimal": 24.0, "rain_optimal": 900.0}
                },
                "wheat": {
                    "cost": {"seed": 30.0, "fertilizer": 50.0, "pesticide": 20.0, "labor": 100.0},  # Total: 200
                    "market": {"price": 500.0, "volatility": 0.1},
                    "yield": {"base": 4.5, "temp_optimal": 20.0, "rain_optimal": 600.0}
                }
            }
        }
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_params = json.load(f)
                    print(f"Loaded raw params: {loaded_params}")
                    # Map top-level simulation parameters
                    if any(k in loaded_params for k in ["years", "initial_savings", "total_land"]):
                        self.params["simulation"] = {
                            "years": loaded_params.get("years", self.params["simulation"]["years"]),
                            "initial_savings": loaded_params.get("initial_savings", self.params["simulation"]["initial_savings"]),
                            "total_land": loaded_params.get("total_land", self.params["simulation"]["total_land"])
                        }
                    # Map env_params to environment
                    if "env_params" in loaded_params:
                        self.params["environment"] = loaded_params.get("env_params", self.params["environment"])
                    # Handle crops
                    if "crops" in loaded_params:
                        if isinstance(loaded_params["crops"], list):
                            print(f"Converting crops list: {loaded_params['crops']}")
                            self.params["crops"] = {}
                            for crop in loaded_params["crops"]:
                                self.params["crops"][crop] = {
                                    "cost": loaded_params.get("cost_params", {}).get(crop, {
                                        "seed": 30.0, "fertilizer": 50.0, "pesticide": 20.0, "labor": 100.0
                                    }),
                                    "market": loaded_params.get("market_params", {}).get(crop, {
                                        "price": 500.0, "volatility": 0.1
                                    }),
                                    "yield": {"base": 4.5, "temp_optimal": 25.0, "rain_optimal": 800.0}
                                }
                        elif isinstance(loaded_params["crops"], dict):
                            self.params["crops"] = loaded_params["crops"]
                    print(f"Processed params: {self.params}")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to load {config_file}: {e}. Using default parameters.")
        else:
            print(f"Config file {config_file} not found. Using default parameters.")
        self.crops = list(self.params["crops"].keys())
        self.num_crops = len(self.crops)

    def get_crop_params(self, crop_name):
        return self.params["crops"].get(crop_name, {})

    def get_crop_market_params(self, crop_name):
        return self.params["crops"].get(crop_name, {}).get("market", {"price": 500.0, "volatility": 0.1})

    def add_crop(self, crop_name, cost=None, market=None, crop_yield=None):
        if crop_name in self.params["crops"]:
            return
        self.params["crops"][crop_name] = {
            "cost": cost or {"seed": 30.0, "fertilizer": 50.0, "pesticide": 20.0, "labor": 100.0},
            "market": market or {"price": 500.0, "volatility": 0.1},
            "yield": crop_yield or {"base": 4.5, "temp_optimal": 25.0, "rain_optimal": 800.0}
        }
        self.crops.append(crop_name)
        self.num_crops = len(self.crops)

    def remove_crop(self, crop_name):
        if crop_name in self.params["crops"]:
            del self.params["crops"][crop_name]
            self.crops.remove(crop_name)
            self.num_crops = len(self.crops)

    def save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.params, f, indent=4)

    def copy(self):
        new_config = Config()
        new_config.params = deepcopy(self.params)
        new_config.crops = list(self.params["crops"].keys())
        new_config.num_crops = len(new_config.crops)
        return new_config

    def validate(self):
        if not self.crops:
            raise ValueError("No crops defined in configuration")
        for crop in self.crops:
            if crop not in self.params["crops"]:
                raise ValueError(f"Crop {crop} not found in params")
            for key in ["cost", "market", "yield"]:
                if key not in self.params["crops"][crop]:
                    raise ValueError(f"Missing {key} parameters for crop {crop}")
        if self.params["simulation"]["total_land"] <= 0:
            raise ValueError("Total land must be positive")
        if self.params["simulation"]["years"] <= 0:
            raise ValueError("Simulation years must be positive")

    @property
    def total_land(self):
        return self.params["simulation"]["total_land"]

    @property
    def initial_savings(self):
        return self.params["simulation"]["initial_savings"]

    @property
    def years(self):
        return self.params["simulation"]["years"]

    @property
    def env_params(self):
        return self.params["environment"]
