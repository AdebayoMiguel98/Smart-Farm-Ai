import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agri_rl_optimizer/src')))
import streamlit as st
from src.config import Config
from src.environment import AgriEnv
from src.rl_model import RLModel

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment import AgriEnv
# ... rest of your imports





def train_default_model():
    default_params = {
        "years": 20,
        "initial_savings": 10000.0,
        "total_land": 100.0,
        "crops": ["wheat", "maize", "rice", "soybeans"],
        "env_params": {"temperature": 25.0, "rainfall": 1000.0},
        "cost_params": {crop: {"seed": 50.0, "fertilizer": 100.0, "pesticide": 30.0, "labor": 200.0} for crop in ["wheat", "maize", "rice", "soybeans"]},
        "market_params": {crop: {"price": 500.0, "volatility": 0.1} for crop in ["wheat", "maize", "rice", "soybeans"]}
    }
    os.makedirs("data", exist_ok=True)
    with open("data/default_params.json", "w") as f:
        import json
        json.dump(default_params, f, indent=4)
    config = Config()
    env = AgriEnv(config)
    model = RLModel(env)
    model.train(total_timesteps=10000)

if not os.path.exists("data/rl_model.zip"):
    train_default_model()

os.system("streamlit run src/streamlit_app.py")