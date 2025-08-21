# tests/test_load_model.py
import torch
import numpy as np
from src.config import Config
from src.environment import AgriEnv
from src.rl_model import RLModel

def test_load_and_forward(path="data/rl_model.pth"):
    cfg = Config()
    env = AgriEnv(cfg)
    model = RLModel(env)
    try:
        model.load(path)
    except Exception as e:
        print("Load failed:", e)
        raise

    # check actor param count > 0
    actor_params = sum(p.numel() for p in model.actor.parameters())
    critic_params = sum(p.numel() for p in model.critic.parameters())
    print(f"Actor params: {actor_params}, Critic params: {critic_params}")

    # create a dummy state and run actor
    state = env.reset()[0]  # numpy array
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)  # batch dim
    with torch.no_grad():
        action = model.actor(state_tensor)
    print("Actor forward produced action shape:", action.shape)

    # create dummy action and run critic
    # action must match action_dim
    action_tensor = torch.zeros((1, model.action_dim), dtype=torch.float32).to(model.device)
    with torch.no_grad():
        q = model.critic(state_tensor, action_tensor)
    print("Critic forward produced q shape:", q.shape)

    print("Model load + forward test passed.")

if __name__ == "__main__":
    test_load_and_forward()
