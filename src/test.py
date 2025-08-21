from config import Config
from environment import AgriEnv
from rl_model import RLModel

cfg = Config()
env = AgriEnv(cfg)
m = RLModel(env)
m.load("data/rl_model.pth")
print("Loaded OK. Actor params:", sum(p.numel() for p in m.actor.parameters()))
