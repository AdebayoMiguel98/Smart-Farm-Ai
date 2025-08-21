agri_rl_optimizer/
├── src/
│   ├── __init__.py          # Makes src a package
│   ├── config.py            # Handles parameter loading/validation/copy
│   ├── environment.py       # Custom Gymnasium environment for farming simulation
│   ├── rl_model.py          # RL model training/testing (DDPG/PPO)
│   ├── simulation.py        # Runs RL and baseline simulations on instances
│   ├── results.py           # Aggregates results and prepares for display/export
│   ├── streamlit_app.py     # Streamlit web app for input, flow, outputs
├── data/
│   ├── default_params.json  # Default configuration with 20 crops
├── tests/
│   ├── test_config.py       # Unit tests for config
│   ├── test_environment.py  # Unit tests for environment
├── main.py                  # Entry point to launch Streamlit app
├── requirements.txt         # Dependencies pinned for Python 3.10.4

├── README.md                # Project description, setup instructions
