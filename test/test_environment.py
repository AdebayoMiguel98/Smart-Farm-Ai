
import pytest
from src.config import Config
from src.environment import AgriEnv
import numpy as np

def test_environment():
    config = Config()
    env = AgriEnv(config)
    obs, _ = env.reset()
    assert obs.shape == (len(config.crops) + 3,)  # Savings + allocations + temp + rainfall
    action = np.array([25.0] * len(config.crops))
    obs, reward, terminated, truncated, _ = env.step(action)
    assert np.isclose(sum(env.land_allocation), config.total_land, atol=1e-5)
    assert not terminated  # Not done after one step
    assert not truncated
