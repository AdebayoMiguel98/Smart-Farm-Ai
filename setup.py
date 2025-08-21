from setuptools import setup, find_packages

setup(
    name="agri_rl_optimizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "numpy",
        "pandas",
        "gymnasium",
        "stable-baselines3",
        "plotly",
        "fpdf"
    ],
)