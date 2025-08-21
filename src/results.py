import numpy as np
import pandas as pd
import os

class Results:
    def __init__(self, config):
        self.config = config

    def plot_results(self, results, scenario_name):
        # No Matplotlib calls; plotting handled by Streamlit
        pass

    def analyze_policy(self, results):
        # Analyze if policy favors profitable crops
        avg_profits = {}
        for i, crop in enumerate(self.config.crops):
            crop_profits = [
                (results["profits"][t] / results["allocations"][t][i] if results["allocations"][t][i] > 0 else 0)
                for t in range(len(results["years"]))
            ]
            avg_profits[crop] = np.mean([p for p in crop_profits if p != 0])
        return avg_profits