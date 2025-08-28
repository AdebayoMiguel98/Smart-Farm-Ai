import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import tempfile
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Debug import issue
try:
    from src.simulation import Simulation
except ImportError as e:
    st.error(f"Failed to import Simulation from src.simulation: {str(e)}")
    sim_path = os.path.join(PROJECT_ROOT, "src", "simulation.py")
    if os.path.exists(sim_path):
        with open(sim_path, "r") as f:
            st.text(f"Contents of simulation.py:\n{f.read()}")
    else:
        st.error(f"Simulation.py not found at {sim_path}")
    st.stop()

from src.config import Config
from src.environment import AgriEnv

# Page Configuration
st.set_page_config(
    page_title="🌾 Agricultural RL Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Config and Session State
config = Config()
if 'crops' not in st.session_state:
    st.session_state.crops = {}

# Sidebar
with st.sidebar:
    st.header("🌱 Simulation Parameters")

    with st.expander("1. Basic Parameters", expanded=True):
        config.params["simulation"]["years"] = st.slider(
            "Simulation Years",
            min_value=1,
            max_value=50,
            value=20,
            key="years_slider"
        )
        config.params["simulation"]["initial_savings"] = st.number_input(
            "Initial Savings ($)",
            min_value=0.0,
            value=10000.0,
            key="initial_savings"
        )
        config.params["simulation"]["total_land"] = st.number_input(
            "Total Land (ha)",
            min_value=0.1,
            value=100.0,
            key="total_land"
        )

    with st.expander("2. Environmental Factors", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            config.params["environment"]["temperature"] = st.slider(
                "Avg Temperature (°C)",
                min_value=0.0,
                max_value=40.0,
                value=25.0,
                key="temperature"
            )
        with col2:
            config.params["environment"]["rainfall"] = st.slider(
                "Avg Rainfall (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=1000.0,
                key="rainfall"
            )

    # Crop Management Section
    st.subheader("3. Crop Management")
    # Add new crop form
    with st.form("add_crop_form"):
        st.subheader("Add New Crop")
        new_crop_name = st.text_input("Crop Name", key="new_crop_name").strip()
        if new_crop_name and new_crop_name not in st.session_state.crops:
            st.markdown("**Cost Parameters ($/ha)**")
            cols_cost = st.columns(4)
            with cols_cost[0]:
                seed_cost = st.number_input("Seed", min_value=0.0, value=50.0, key="new_seed")
            with cols_cost[1]:
                fertilizer_cost = st.number_input("Fertilizer", min_value=0.0, value=100.0, key="new_fertilizer")
            with cols_cost[2]:
                pesticide_cost = st.number_input("Pesticide", min_value=0.0, value=30.0, key="new_pesticide")
            with cols_cost[3]:
                labor_cost = st.number_input("Labor", min_value=0.0, value=200.0, key="new_labor")

            st.markdown("**Market Parameters**")
            cols_market = st.columns(2)
            with cols_market[0]:
                base_price = st.number_input("Base Price ($/ton)", min_value=0.0, value=500.0, key="new_price")
            with cols_market[1]:
                volatility = st.slider("Price Volatility", min_value=0.0, max_value=1.0, value=0.1, key="new_volatility")

            st.markdown("**Yield Parameters**")
            cols_yield = st.columns(3)
            with cols_yield[0]:
                base_yield = st.number_input("Base Yield (ton/ha)", min_value=0.0, value=4.5, key="new_base_yield")
            with cols_yield[1]:
                temp_optimal = st.number_input("Optimal Temperature (°C)", min_value=0.0, value=25.0, key="new_temp_optimal")
            with cols_yield[2]:
                rain_optimal = st.number_input("Optimal Rainfall (mm)", min_value=0.0, value=800.0, key="new_rain_optimal")

        if st.form_submit_button("Add Crop"):
            st.session_state.crops[new_crop_name] = {
                "cost": {"seed": seed_cost, "fertilizer": fertilizer_cost, "pesticide": pesticide_cost, "labor": labor_cost},
                "market": {"price": base_price, "volatility": volatility},
                "yield": {"base": base_yield, "temp_optimal": temp_optimal, "rain_optimal": rain_optimal}
            }
            st.success(f"Added {new_crop_name}")
            st.experimental_rerun()

    # Display and manage existing crops
    st.subheader("Current Crops")
    if st.session_state.crops:
        for crop in sorted(list(st.session_state.crops.keys())):
            with st.expander(f"{crop} (Edit/Remove)", expanded=False):
                cols_cost = st.columns(4)
                with cols_cost[0]:
                    seed_cost = st.number_input("Seed", min_value=0.0, value=st.session_state.crops[crop]["cost"]["seed"], key=f"{crop}_seed")
                with cols_cost[1]:
                    fertilizer_cost = st.number_input("Fertilizer", min_value=0.0, value=st.session_state.crops[crop]["cost"]["fertilizer"], key=f"{crop}_fertilizer")
                with cols_cost[2]:
                    pesticide_cost = st.number_input("Pesticide", min_value=0.0, value=st.session_state.crops[crop]["cost"]["pesticide"], key=f"{crop}_pesticide")
                with cols_cost[3]:
                    labor_cost = st.number_input("Labor", min_value=0.0, value=st.session_state.crops[crop]["cost"]["labor"], key=f"{crop}_labor")

                cols_market = st.columns(2)
                with cols_market[0]:
                    base_price = st.number_input("Base Price ($/ton)", min_value=0.0, value=st.session_state.crops[crop]["market"]["price"], key=f"{crop}_price")
                with cols_market[1]:
                    volatility = st.slider("Price Volatility", min_value=0.0, max_value=1.0, value=st.session_state.crops[crop]["market"]["volatility"], key=f"{crop}_volatility")

                cols_yield = st.columns(3)
                with cols_yield[0]:
                    base_yield = st.number_input("Base Yield (ton/ha)", min_value=0.0, value=st.session_state.crops[crop]["yield"]["base"], key=f"{crop}_yield_base")
                with cols_yield[1]:
                    temp_optimal = st.number_input("Optimal Temperature (°C)", min_value=0.0, value=st.session_state.crops[crop]["yield"]["temp_optimal"], key=f"{crop}_yield_temp")
                with cols_yield[2]:
                    rain_optimal = st.number_input("Optimal Rainfall (mm)", min_value=0.0, value=st.session_state.crops[crop]["yield"]["rain_optimal"], key=f"{crop}_yield_rain")

                if st.button(f"Update {crop}"):
                    st.session_state.crops[crop] = {
                        "cost": {"seed": seed_cost, "fertilizer": fertilizer_cost, "pesticide": pesticide_cost, "labor": labor_cost},
                        "market": {"price": base_price, "volatility": volatility},
                        "yield": {"base": base_yield, "temp_optimal": temp_optimal, "rain_optimal": rain_optimal}
                    }
                    st.success(f"Updated {crop}")
                    st.experimental_rerun()

                if st.button(f"Remove {crop}"):
                    del st.session_state.crops[crop]
                    st.success(f"Removed {crop}")
                    st.experimental_rerun()

    selected_crops = sorted(list(st.session_state.crops.keys()))

    with st.expander("7. Configuration", expanded=False):
        if st.button("💾 Save Current Configuration"):
            config.params["crops"] = st.session_state.crops
            config.save_to_file("data/current_config.json")
            st.success("Configuration saved!")

    with st.expander("8. Model Management", expanded=False):
        if st.button("🔄 Retrain Model"):
            st.session_state.retrain_model = True
            st.experimental_rerun()

    st.divider()
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        st.session_state.run_simulation = True
    else:
        st.session_state.run_simulation = False

# Main Content Section
st.title("🌾 Agricultural RL Optimizer")
st.markdown("""
This tool uses Reinforcement Learning to optimize crop land allocation for smallholder farmers.
Configure your parameters in the sidebar and run the simulation to see optimal strategies.
""")

if not selected_crops:
    st.warning("Please add at least one crop to include in the simulation")
    st.stop()

# Initialize session state
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
if 'retrain_model' not in st.session_state:
    st.session_state.retrain_model = False

if st.session_state.run_simulation or st.session_state.retrain_model:
    # Simulation Execution
    with st.spinner("Running simulation... This may take several minutes"):
        try:
            # Debug: Verify Config
            print(f"Config params: {config.params}")
            config.validate()

            # Update config with dynamic crops
            config.params["crops"] = st.session_state.crops
            config.crops = selected_crops
            config.num_crops = len(selected_crops)

            # Initialize environment and model
            env = AgriEnv(config)
            simulation = Simulation(env, config)

            # Retrain model if requested or if it doesn't exist
            if st.session_state.retrain_model or not os.path.exists("data/rl_model.pth"):
                print("Training RL model with current crops...")
                simulation.model.train(total_timesteps=20000)
                simulation.model.save()
                st.session_state.retrain_model = False  # Reset after retraining

            # Run 100 instances for RL and baseline
            rl_results = simulation.run_multiple(num_episodes=100)
            baseline_results = simulation.run_baseline(num_episodes=100)

            # Save results
            simulation.save_results(rl_results, "rl_results")
            simulation.save_results(baseline_results, "baseline_results")

            st.session_state.simulation_results = {
                "rl": rl_results,
                "baseline": baseline_results,
                "config": config.copy(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.stop()

    results = st.session_state.simulation_results

    # Results Visualization
    st.success("✅ Simulation completed successfully!")
    st.caption(f"Generated at {results['timestamp']}")

    # 1. Key Metrics Comparison
    st.header("📊 Performance Summary")
    col1, col2, col3 = st.columns(3)

    rl_final_savings = results["rl"]["savings"][-1]
    baseline_final_savings = results["baseline"]["savings"][-1]
    improvement = ((rl_final_savings - baseline_final_savings) / baseline_final_savings * 100) if baseline_final_savings != 0 else 0

    with col1:
        st.metric(
            "Average Final Savings (RL)",
            f"${rl_final_savings:,.2f}",
            delta=f"{improvement:.1f}% vs Baseline"
        )

    with col2:
        st.metric(
            "Average Final Savings (Baseline)",
            f"${baseline_final_savings:,.2f}"
        )

    with col3:
        st.metric(
            "Total Profit (RL)",
            f"${sum(results['rl']['profits']):,.2f}"
        )

    # 2. Average Final Savings Across Parameter Sets
    st.header("📊 Average Final Savings Across Parameter Sets")
    param_sets = [
        {"initial_savings": 5000.0, "label": "Low Savings ($5K)"},
        {"initial_savings": 10000.0, "label": "Default Savings ($10K)"},
        {"initial_savings": 20000.0, "label": "High Savings ($20K)"}
    ]

    param_results = []
    for param in param_sets:
        temp_config = config.copy()
        temp_config.params["simulation"]["initial_savings"] = param["initial_savings"]
        temp_config.params["crops"] = st.session_state.crops
        temp_config.crops = selected_crops
        temp_config.num_crops = len(selected_crops)
        temp_env = AgriEnv(temp_config)
        temp_simulation = Simulation(temp_env, temp_config)
        temp_results = temp_simulation.run_multiple(num_episodes=100)
        param_results.append({
            "Parameter Set": param["label"],
            "Avg Final Savings": temp_results["savings"][-1]
        })

    param_df = pd.DataFrame(param_results)
    fig_param = px.bar(
        param_df,
        x="Parameter Set",
        y="Avg Final Savings",
        title="Average Final Savings by Parameter Set (100 Instances)",
        labels={"Avg Final Savings": "Savings ($)"}
    )
    fig_param.update_layout(
        showlegend=False,
        xaxis_title="Parameter Set",
        yaxis_title="Savings ($)",
        title_x=0.5
    )
    st.plotly_chart(fig_param, use_container_width=True)

    # 3. Detailed Visualization Results
    st.header("📈 Detailed Visualization Results")

    # 3.1 How Land Allocation Changes Over Time
    st.subheader("🌱 How Land Allocation Changes Over Time")
    alloc_data = []
    if not results["rl"]["allocations"] or len(results["rl"]["allocations"][0]) != len(selected_crops):
        st.error("No valid allocation data available. Check simulation output or selected crops.")
        st.write("Debug: RL Allocations", results["rl"]["allocations"])
        st.write("Debug: Selected Crops", selected_crops)
    else:
        for year, yearly_alloc in enumerate(results["rl"]["allocations"], 1):
            total_alloc = sum(yearly_alloc)
            if total_alloc == 0:
                st.write(f"Debug: Year {year} allocations sum to 0: {yearly_alloc}")
            for crop_idx, alloc in enumerate(yearly_alloc):
                if crop_idx < len(selected_crops):
                    alloc_data.append({
                        "Year": year,
                        "Crop": selected_crops[crop_idx],
                        "Allocation (ha)": alloc,
                        "Percentage": (alloc / config.params["simulation"]["total_land"]) * 100
                    })

    alloc_df = pd.DataFrame(alloc_data)
    st.write("Debug: Allocation DataFrame", alloc_df)

    if not alloc_df.empty:
        fig_alloc = px.area(
            alloc_df,
            x="Year",
            y="Allocation (ha)",
            color="Crop",
            title="Optimal Land Allocation by Year (RL Policy)",
            labels={"Allocation (ha)": "Land Allocation (hectares)"},
            animation_frame="Year" if len(alloc_df["Year"].unique()) > 1 else None,
            range_y=[0, config.params["simulation"]["total_land"]]
        )
        fig_alloc.update_layout(
            showlegend=True,
            xaxis_title="Year",
            yaxis_title="Land Allocation (hectares)",
            title_x=0.5
        )
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.warning("Allocation data is empty. Check RL model or simulation configuration.")

    # 3.2 How Savings Increase or Decrease Over Time
    st.subheader("💰 How Savings Increase or Decrease Over Time")
    savings_data = []
    for year in range(config.params["simulation"]["years"]):
        savings_data.append({
            "Year": year + 1,
            "Savings": results["rl"]["savings"][year],
            "Strategy": "RL Policy"
        })
        savings_data.append({
            "Year": year + 1,
            "Savings": results["baseline"]["savings"][year],
            "Strategy": "Naive Baseline"
        })

    savings_df = pd.DataFrame(savings_data)
    fig_savings = px.line(
        savings_df,
        x="Year",
        y="Savings",
        color="Strategy",
        line_dash="Strategy",
        title="Savings Growth: RL vs Naive Baseline",
        labels={"Savings": "Savings ($)"}
    )
    max_rl_savings = max(results["rl"]["savings"])
    max_rl_year = results["rl"]["savings"].index(max_rl_savings) + 1
    min_rl_savings = min(results["rl"]["savings"])
    min_rl_year = results["rl"]["savings"].index(min_rl_savings) + 1
    fig_savings.add_annotation(
        x=max_rl_year, y=max_rl_savings,
        text=f"Max RL: ${max_rl_savings:,.2f}",
        showarrow=True, arrowhead=2
    )
    fig_savings.add_annotation(
        x=min_rl_year, y=min_rl_savings,
        text=f"Min RL: ${min_rl_savings:,.2f}",
        showarrow=True, arrowhead=2
    )
    fig_savings.update_layout(
        showlegend=True,
        xaxis_title="Year",
        yaxis_title="Savings ($)",
        title_x=0.5
    )
    st.plotly_chart(fig_savings, use_container_width=True)

    # 3.3 Which Crops Performed Best
    st.subheader("📊 Which Crops Performed Best")
    crop_profits = []
    for crop in selected_crops:
        total_profit = sum(results["rl"]["crop_profits"][crop])
        crop_profits.append({
            "Crop": crop,
            "Total Profit": total_profit
        })

    profit_df = pd.DataFrame(crop_profits)
    profit_df = profit_df.sort_values(by="Total Profit", ascending=False)
    fig_profit = px.pie(
        profit_df,
        names="Crop",
        values="Total Profit",
        title="Total Profit Contribution by Crop (RL Policy)",
        labels={"Total Profit": "Total Profit ($)"}
    )
    fig_profit.update_layout(
        showlegend=True,
        title_x=0.5
    )
    st.plotly_chart(fig_profit, use_container_width=True)

    # 4. Profit Comparison
    st.header("💸 Profit Comparison: RL vs Naive Baseline")
    profit_data = []
    for year in range(config.params["simulation"]["years"]):
        profit_data.append({
            "Year": year + 1,
            "Profit": results["rl"]["profits"][year],
            "Strategy": "RL Policy"
        })
        profit_data.append({
            "Year": year + 1,
            "Profit": results["baseline"]["profits"][year],
            "Strategy": "Naive Baseline"
        })

    profit_df = pd.DataFrame(profit_data)
    fig_profit_compare = px.line(
        profit_df,
        x="Year",
        y="Profit",
        color="Strategy",
        line_dash="Strategy",
        title="Profit Comparison: RL vs Naive Baseline",
        labels={"Profit": "Profit ($)"}
    )
    fig_profit_compare.update_layout(
        showlegend=True,
        xaxis_title="Year",
        yaxis_title="Profit ($)",
        title_x=0.5
    )
    st.plotly_chart(fig_profit_compare, use_container_width=True)

    # 5. Detailed Annual Results Table
    st.header("📋 Detailed Annual Results")
    detailed_data = []
    for year in range(config.params["simulation"]["years"]):
        yearly_data = {
            "Year": year + 1,
            "Avg Profit ($)": results["rl"]["profits"][year],
            "Avg Savings ($)": results["rl"]["savings"][year]
        }
        for crop_idx, crop in enumerate(selected_crops):
            yearly_data[f"{crop} (ha)"] = results["rl"]["allocations"][year][crop_idx]
            yearly_data[f"{crop} Profit ($)"] = results["rl"]["crop_profits"][crop][year]
        detailed_data.append(yearly_data)

    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(
        detailed_df,
        use_container_width=True,
        hide_index=True
    )

    # 6. Summary Statistics Table
    st.header("📊 Summary Statistics")
    summary_data = {
        "Metric": ["Final Savings (RL)", "Final Savings (Baseline)", "Total Profit (RL)", "Total Profit (Baseline)"],
        "Mean": [
            results["rl"]["savings"][-1],
            results["baseline"]["savings"][-1],
            sum(results["rl"]["profits"]),
            sum(results["baseline"]["profits"])
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # 7. Export Options
    st.header("📤 Export Results")

    # CSV Export
    combined_csv = pd.concat([
        detailed_df.assign(Table="Detailed Annual Results"),
        summary_df.assign(Table="Summary Statistics")
    ])
    csv = combined_csv.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="agricultural_rl_results.csv",
        mime="text/csv"
    )

    # PDF Report Generation
    if st.button("🖨️ Generate PDF Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title and Metadata
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Agricultural RL Optimization Report", ln=1, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 10, txt=f"Generated on {results['timestamp']}", ln=1, align='C')

            # Summary Statistics
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Summary Statistics", ln=1)
            pdf.set_font("Arial", '', 12)
            for _, row in summary_df.iterrows():
                pdf.cell(200, 10, txt=f"{row['Metric']}: Mean=${row['Mean']:,.2f}", ln=1)

            # Parameter Set Results
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Average Final Savings by Parameter Set", ln=1)
            pdf.set_font("Arial", '', 12)
            for _, row in param_df.iterrows():
                pdf.cell(200, 10, txt=f"{row['Parameter Set']}: Avg Savings=${row['Avg Final Savings']:,.2f}", ln=1)

            # Detailed Annual Results
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Detailed Annual Results", ln=1)
            pdf.set_font("Arial", '', 12)
            for _, row in detailed_df.iterrows():
                line = f"Year {int(row['Year'])}: "
                for crop in selected_crops:
                    if f"{crop} (ha)" in row:
                        line += f"{crop}={row[f'{crop} (ha)']:.2f}ha, "
                    if f"{crop} Profit ($)" in row:
                        line += f"{crop} Profit=${row[f'{crop} Profit ($)']:.2f}, "
                line += f"Profit=${row['Avg Profit ($)']:.2f}, Savings=${row['Avg Savings ($)']:.2f}"
                pdf.cell(200, 10, txt=line, ln=1)

            pdf.output(tmpfile.name)

            with open(tmpfile.name, "rb") as f:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=f,
                    file_name="agricultural_rl_report.pdf",
                    mime="application/pdf"
                )