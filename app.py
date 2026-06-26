import numpy as np
import pandas as pd
import streamlit as st
import graphviz
import os
from model import run_hybrid_pipeline  # Core PSO-BPNN calculation loop

# --- PROFESSIONAL EXECUTIVE STYLING CONFIGURATIONS ---
st.set_page_config(
    page_title="PSO-BPNN Weather Optimization Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom injection to refine visual padding and modern elements
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        .stButton button { width: 100%; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SYSTEM GRAPHVIZ ARCHITECTURE GENERATOR FUNCTIONS ---
def generate_pipeline_flowchart():
    dot = graphviz.Digraph(comment='Data Pipeline Map')
    dot.attr(rankdir='LR', size='16,8!', ratio='fill', bgcolor='transparent')
    dot.attr('node', shape='box', style='filled,rounded', color='#2B6CB0', 
             fontcolor='white', fontname='Helvetica-Bold', fontsize='14', height='1.2', width='2.6')
    
    dot.node('A', 'Weather Input Vector\n(Uploaded File / Default CSV)')
    dot.node('B', 'Data Processing Matrix\n(MinMax Normalization)')
    
    dot.attr('node', color='#2F855A')
    dot.node('C', 'Particle Swarm\nOptimization Engine')
    dot.node('D', 'Optimized Feature\nSubspace Array')
    
    dot.attr('node', color='#C53030')
    dot.node('E', 'Deep Backpropagation\nNeural Network')
    dot.node('F', 'High-Precision\nWeather Projection')
    
    dot.attr('edge', fontname='Helvetica', fontsize='11', penwidth='2.0', color='#4A5568')
    dot.edge('A', 'B')
    dot.edge('B', 'C', label=' Master Attributes')
    dot.edge('B', 'D', style='dashed', label=' Spatial Filter')
    dot.edge('C', 'D', label=' Global Best Vector')
    dot.edge('D', 'E', label=' Refined Matrix')
    dot.edge('E', 'F')
    return dot

def generate_algo_flowchart(selected_feats):
    dot = graphviz.Digraph(comment='PSO Optimization Iterative Loop')
    dot.attr(rankdir='TB', size='14,12!', ratio='fill', bgcolor='transparent')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', height='1.0', width='3.0')
    dot.attr('edge', penwidth='1.8', color='#718096')
    
    dot.node('Start', 'Initialize Swarm Matrix\n(Random Binary Coordinates)', shape='ellipse', color='#4A5568', fontcolor='white')
    dot.node('Fit', 'Evaluate Swarm Fitness\nCost = f(MSE, Feature Dimension Count)', shape='box', color='#2B6CB0', fontcolor='white')
    dot.node('Update', 'Adjust Velocity & Coordinate Vectors\n(Sigmoid Mapping Filter Loop)', shape='box', color='#2B6CB0', fontcolor='white')
    dot.node('Check', 'Convergence Met?', shape='diamond', color='#D69E2E', fontcolor='black', height='1.5')
    
    features_str = "\\n".join(selected_feats[:5]) + ("\\n..." if len(selected_feats) > 5 else "")
    dot.node('End', f'Optimal Features Input Matrix:\\n{features_str}', shape='parallelogram', color='#2F855A', fontcolor='white', width='3.8')
    
    dot.edge('Start', 'Fit')
    dot.edge('Fit', 'Update')
    dot.edge('Update', 'Check')
    dot.edge('Check', 'Fit', label=' No (Next Iteration)')
    dot.edge('Check', 'End', label=' Yes (Optimized Matrix)')
    return dot

# ==========================================
# HEADER ACTION SECTION
# ==========================================
st.write("## 🌦️ Hybrid Intelligent Predictive System Environment")
st.caption("**Research Project Direct Title Alignment:** Development of a Weather Forecasting Prediction System Using Particle Swarm Optimization (PSO) Feature Selection and Backpropagation Neural Network (BPNN) Algorithms")
st.write("---")

# ==========================================
# SIDEBAR CONTROL CONSOLE ROOM
# ==========================================
st.sidebar.markdown("### 🛠️ Control Room Console")
st.sidebar.markdown("---")

# 1. Dataset Management Space
st.sidebar.markdown("#### 📂 1. Data Source Handling")
uploaded_file = st.sidebar.file_uploader("Ingest Experimental CSV File", type=["csv"], help="Upload an external weather records file matrix.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "weather_prediction_dataset2.csv")

@st.cache_data
def load_data(file_source):
    return pd.read_csv(file_source)

if uploaded_file is not None:
    try:
        df_raw = load_data(uploaded_file)
        st.sidebar.success("✅ Experimental file loaded.")
    except Exception as e:
        st.sidebar.error("❌ Data Parsing Error.")
        st.stop()
else:
    try:
        df_raw = load_data(DEFAULT_CSV_PATH)
        st.sidebar.info("ℹ️ Using default benchmark dataset.")
    except Exception as e:
        st.sidebar.error("❌ Default file location missing.")
        st.stop()

# 2. Variable Configuration Selectors
all_columns = df_raw.columns.tolist()
excluded = ['DATE', 'MONTH', 'date', 'month', 'Id', 'id']
selectable_targets = [col for col in all_columns if col not in excluded]
default_target = "BASEL_temp_mean" if "BASEL_temp_mean" in selectable_targets else selectable_targets[0]

target_column = st.sidebar.selectbox("🎯 Target Variable", options=selectable_targets, index=selectable_targets.index(default_target) if default_target in selectable_targets else 0)

st.sidebar.markdown("---")

# 3. Model Engine Core Hyperparameters Tuning
st.sidebar.markdown("#### ⚙️ 2. Metaheuristic Meta-Params")
swarm_size = st.sidebar.slider("Swarm Population Density", min_value=5, max_value=25, value=10, step=1)
iterations = st.sidebar.slider("PSO Max Generations Loop", min_value=5, max_value=25, value=10, step=1)

st.sidebar.markdown("#### 🧠 3. Neural Topology Layer Layout")
hidden_layers_input = st.sidebar.text_input("Hidden Architecture Struct", value="32, 16", help="Comma-separated node count dimensions per tier.")

try:
    layer_sizes = tuple(map(int, hidden_layers_input.split(',')))
except:
    st.sidebar.error("Syntax Error. Format schema: 32, 16")
    st.stop()

st.sidebar.markdown("---")
# Big Primary Action Trigger Button
execute_pipeline = st.sidebar.button("🚀 EXECUTE PROCESSING PIPELINE")


# ==========================================
# MAIN INTERFACE EXECUTIVE CONTAINER LAYOUT
# ==========================================
if execute_pipeline:
    with st.spinner("Processing Metaheuristic Swarm Trajectories & Network Training Calculations..."):
        try:
            results = run_hybrid_pipeline(
                df=df_raw,
                target_column=target_column,
                excluded_cols=excluded,
                swarm_size=swarm_size,
                pso_iters=iterations,
                bpnn_topology=layer_sizes
            )
        except Exception as e:
            st.error(f"❌ Core Engine Runtime System Failure: {str(e)}")
            st.stop()

    # --- ROW 1: PRIMARY KPI SCORECARD METRICS DISPLAY ---
    st.markdown("### 📈 Core Forecast Validation Scorecard")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # Calculate variations vs baseline to prove accuracy enhancements
    mae_improvement = ((results['base_mae'] - results['hybrid_mae']) / results['base_mae']) * 100
    r2_improvement = results['hybrid_r2'] - results['base_r2']
    compression_ratio = (1 - (len(results['pso_selected_features']) / results['total_raw_features'])) * 100

    metric_col1.metric("System Mean Absolute Error", f"{results['hybrid_mae']:.4f}", f"-{mae_improvement:.1f}% Delta Error", delta_color="inverse")
    metric_col2.metric("System Root Mean Square Error", f"{results['hybrid_rmse']:.4f}")
    metric_col3.metric("System R² Fit Metric", f"{results['hybrid_r2']:.4f}", f"+{r2_improvement:.3f} Gain")
    metric_col4.metric("Dataset Dimensionality Filtered", f"{compression_ratio:.1f}%", f"-{results['total_raw_features'] - len(results['pso_selected_features'])} Unused Features", delta_color="off")
    st.write("---")

    # --- ROW 2: SPLIT FRAME DETAILS AND GRAPHICAL ROADMAP DATA ---
    left_panel, right_panel = st.columns([1, 1])

    with left_panel:
        st.markdown("#### 📉 Analytical Model Error Variance Chart")
        chart_dataframe = pd.DataFrame({
            "PSO-BPNN Proposed Model": [results['hybrid_mae'], results['hybrid_rmse']],
            "Standard Baseline Tracker": [results['base_mae'], results['base_rmse']]
        }, index=["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"])
        st.bar_chart(chart_dataframe, use_container_width=True)
        
        st.markdown("#### 🔍 Feature Reduction Processing Data Matrix")
        st.write(f"🧬 **Optimal Predictor Attribute Matrix Subspace Chosen:** ({len(results['pso_selected_features'])} features total)")
        st.info(f", ".join(results['pso_selected_features']))

    with right_panel:
        st.markdown("#### 📋 Operational Forecast Validation Records")
        comparison_table_data = pd.DataFrame({
            "Observed Ground Truth": results['y_test_observed'],
            "PSO-BPNN Hybrid Prediction": results['hybrid_predictions'],
            "Baseline Architecture Trace": results['baseline_predictions'],
            "Calculated Delta Error Residuals": results['y_test_observed'] - results['hybrid_predictions']
        })
        st.dataframe(comparison_table_data.head(100), use_container_width=True, height=365)

    st.write("---")
    
    # --- ROW 3: RESTRUCTURED FLOWCHART LAYOUT AT THE BASE ---
    st.markdown("#### 🗺️ Process Workflow & Structural Engineering Maps")
    flow_tab1, flow_tab2 = st.tabs(["1. End-To-End System Architecture Data Flow", "2. Metaheuristic Evolutionary Swarm Optimization Loop"])
    
    with flow_tab1:
        st.graphviz_chart(generate_pipeline_flowchart(), use_container_width=True)
    with flow_tab2:
        st.graphviz_chart(generate_algo_flowchart(results['pso_selected_features']), use_container_width=True)

else:
    # Clean, distraction-free corporate home view state placeholder
    st.info("💡 Adjust your target parameters inside the sidebar control center panel on the left and select 'Execute Processing Pipeline' to compute structural optimization sequences and view metric graphics.")
    
    st.markdown("#### 📊 Exploratory Dataset Preview Matrix")
    st.dataframe(df_raw.head(10), use_container_width=True)