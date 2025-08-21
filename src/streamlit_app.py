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
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Debug import issue
try:
    from src.simulation import Simulation
except ImportError as e:
    st.error(f"Failed to import Simulation from src.simulation: {str(e)}")
    with open(os.path.join(PROJECT_ROOT, "src", "simulation.py"), "r") as f:
        st.text(f"Contents of simulation.py:\n{f.read()}")
    st.stop()

from src.config import Config
from src.environment import AgriEnv
from src.rl_model import RLModel

# Page Configuration
st.set_page_config(
    page_title="🌾 Agricultural RL Optimizer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Config
config = Config()

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
    
    with st.expander("3. Crop Management", expanded=False):
        available_crops = config.crops
        selected_crops = st.multiselect(
            "Select Crops to Include",
            options=available_crops,
            default=available_crops[:4],
            key="crop_selection"
        )
        add_new_crop = st.checkbox("➕ Add New Crop")
        if add_new_crop:
            with st.form("new_crop_form"):
                new_crop_name = st.text_input("Crop Name").strip()
                st.markdown("**Yield Parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    base_yield = st.number_input(
                        "Base Yield (ton/ha)", 
                        min_value=0.0,
                        value=4.5,
                        key="new_crop_yield"
                    )
                with col2:
                    temp_optimal = st.number_input(
                        "Optimal Temperature (°C)", 
                        min_value=0.0,
                        value=25.0,
                        key="new_crop_temp"
                    )
                with col3:
                    rain_optimal = st.number_input(
                        "Optimal Rainfall (mm)", 
                        min_value=0.0,
                        value=800.0,
                        key="new_crop_rain"
                    )
                if st.form_submit_button("Add Crop") and new_crop_name and new_crop_name not in available_crops:
                    config.add_crop(
                        new_crop_name,
                        crop_yield={"base": base_yield, "temp_optimal": temp_optimal, "rain_optimal": rain_optimal}
                    )
                    st.success(f"Added {new_crop_name} with default parameters")
                    st.experimental_rerun()
        
        if len(selected_crops) > 0 and st.button("❌ Remove Selected Crops"):
            for crop in selected_crops:
                config.remove_crop(crop)
            st.success("Removed selected crops")
            st.experimental_rerun()
    
    with st.expander("4. Cost Parameters ($/ha)", expanded=False):
        for crop in selected_crops:
            st.markdown(f"**{crop.capitalize()}**")
            cols = st.columns(4)
            with cols[0]:
                config.params["crops"][crop]["cost"]["seed"] = st.number_input(
                    "Seed", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["cost"]["seed"],
                    key=f"{crop}_seed"
                )
            with cols[1]:
                config.params["crops"][crop]["cost"]["fertilizer"] = st.number_input(
                    "Fertilizer", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["cost"]["fertilizer"],
                    key=f"{crop}_fertilizer"
                )
            with cols[2]:
                config.params["crops"][crop]["cost"]["pesticide"] = st.number_input(
                    "Pesticide", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["cost"]["pesticide"],
                    key=f"{crop}_pesticide"
                )
            with cols[3]:
                config.params["crops"][crop]["cost"]["labor"] = st.number_input(
                    "Labor", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["cost"]["labor"],
                    key=f"{crop}_labor"
                )
    
    with st.expander("5. Market Parameters", expanded=False):
        for crop in selected_crops:
            st.markdown(f"**{crop.capitalize()}**")
            cols = st.columns(2)
            with cols[0]:
                config.params["crops"][crop]["market"]["price"] = st.number_input(
                    "Base Price ($/ton)", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["market"]["price"],
                    key=f"{crop}_price"
                )
            with cols[1]:
                config.params["crops"][crop]["market"]["volatility"] = st.slider(
                    "Price Volatility", 
                    min_value=0.0,
                    max_value=1.0,
                    value=config.params["crops"][crop]["market"]["volatility"],
                    key=f"{crop}_volatility"
                )
    
    with st.expander("6. Yield Parameters", expanded=False):
        for crop in selected_crops:
            st.markdown(f"**{crop.capitalize()}**")
            cols = st.columns(3)
            with cols[0]:
                config.params["crops"][crop]["yield"]["base"] = st.number_input(
                    "Base Yield (ton/ha)", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["yield"]["base"],
                    key=f"{crop}_yield_base"
                )
            with cols[1]:
                config.params["crops"][crop]["yield"]["temp_optimal"] = st.number_input(
                    "Optimal Temperature (°C)", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["yield"]["temp_optimal"],
                    key=f"{crop}_yield_temp"
                )
            with cols[2]:
                config.params["crops"][crop]["yield"]["rain_optimal"] = st.number_input(
                    "Optimal Rainfall (mm)", 
                    min_value=0.0,
                    value=config.params["crops"][crop]["yield"]["rain_optimal"],
                    key=f"{crop}_yield_rain"
                )
    
    with st.expander("7. Configuration", expanded=False):
        if st.button("💾 Save Current Configuration"):
            config.save_to_file("data/current_config.json")
            st.success("Configuration saved!")
    
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
    st.warning("Please select at least one crop to include in the simulation")
    st.stop()

# Initialize session state
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

if st.session_state.run_simulation:
    # Simulation Execution
    with st.spinner("Running simulation... This may take several minutes"):
        try:
            # Debug: Verify Config
            print(f"Config params: {config.params}")
            config.validate()
            
            # Filter crops to only include selected ones
            config.params["crops"] = {k: v for k, v in config.params["crops"].items() if k in selected_crops}
            config.crops = selected_crops
            config.num_crops = len(selected_crops)
            
            # Initialize environment and model
            env = AgriEnv(config)
            simulation = Simulation(env, config)
            
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
    
    rl_final_savings = results["rl"]["savings"][-1]  # Final averaged savings
    baseline_final_savings = results["baseline"]["savings"][-1]  # Final averaged savings
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
            total_alloc = sum(yearly_alloc)  # Check if sums to 100 ha
            if total_alloc == 0:
                st.write(f"Debug: Year {year} allocations sum to 0: {yearly_alloc}")
            for crop_idx, alloc in enumerate(yearly_alloc):
                if crop_idx < len(selected_crops):  # Ensure index is valid
                    alloc_data.append({
                        "Year": year,
                        "Crop": selected_crops[crop_idx],
                        "Allocation (ha)": alloc,  # Remove max(0) to see raw data
                        "Percentage": (alloc / config.params["simulation"]["total_land"]) * 100
                    })
    
        alloc_df = pd.DataFrame(alloc_data)
        st.write("Debug: Allocation DataFrame", alloc_df)  # Debug output
        
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
    # Add annotations for max/min savings
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