import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from stable_baselines3 import PPO

import sys
# Make sure we can import the environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.env.world_env import WorldEnv

st.set_page_config(page_title="WorldSim Dashboard", layout="wide")

st.title("🌍 WorldSim Dashboard (LIVE)")
st.subheader("Adaptive Resource Scarcity & Political Ideology Simulator")

# --- Initialize Session State ---
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'env' not in st.session_state:
    st.session_state.env = WorldEnv(num_regions=4)
    obs, _ = st.session_state.env.reset()
    st.session_state.obs = obs
    st.session_state.history = []
    
    # Run one step so we have initial info to display
    obs, _, _, _, info = st.session_state.env.step([1, 1, 1, 1])
    st.session_state.obs = obs
    st.session_state.latest_info = info
    
    # Load model once on initialization
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "backend", "training", "saved_model.zip")
    if os.path.exists(model_path):
        st.session_state.model = PPO.load(model_path)
    else:
        # Fallback to a new untrained model to allow simulation if saved model missing
        st.session_state.model = PPO("MlpPolicy", st.session_state.env, verbose=0)
        
if 'simulation_speed' not in st.session_state:
    st.session_state.simulation_speed = 0.5
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("▶️ Start"):
        st.session_state.simulation_running = True
with col2:
    if st.button("⏸️ Pause"):
        st.session_state.simulation_running = False
with col3:
    if st.button("🔄 Reset"):
        st.session_state.simulation_running = False
        st.session_state.step_count = 0
        obs, _ = st.session_state.env.reset()
        st.session_state.obs = obs
        st.session_state.history = []
        # Step once for info
        obs, _, _, _, info = st.session_state.env.step([1, 1, 1, 1])
        st.session_state.obs = obs
        st.session_state.latest_info = info

st.session_state.simulation_speed = st.sidebar.slider("Simulation Speed (seconds per step)", min_value=0.1, max_value=2.0, value=st.session_state.simulation_speed, step=0.1)

# Status Indicator
if st.session_state.simulation_running:
    st.sidebar.success("Simulation is RUNNING")
else:
    st.sidebar.warning("Simulation is PAUSED")

st.sidebar.markdown("---")
st.sidebar.metric("Current Step", st.session_state.step_count)

CLIMATE_MAP = {0: "NORMAL", 1: "DROUGHT", 2: "FLOOD"}
ACTION_MAP = {0: "Conserve", 1: "Balanced", 2: "Aggressive", 3: "Trade", 4: "Tech"}

def plot_radar(ideology):
    categories = ['Cooperation', 'Aggression', 'Sustainability']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ideology,
        theta=categories,
        fill='toself',
        name='Ideology'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig


env = st.session_state.env
info = st.session_state.latest_info
num_regions = env.num_regions

# --- Render UI Top-to-Bottom ---
# Global Metrics
st.markdown("---")
col1, col2, col3 = st.columns(3)
total_res = sum([sum(info["resources"][r][:3]) for r in range(num_regions)])
with col1:
    st.metric("Total Global Resources", f"{total_res:,.1f}")
with col2:
    st.metric("Global Collapse Events", info.get("global_collapses", 0))
with col3:
    st.metric("Current Climate State", CLIMATE_MAP.get(info.get("climate", 0), "UNKNOWN"))
st.markdown("---")
    
# Time Series
st.header("Time-Series Trends")
if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_res = px.line(df, x="Step", y="Total Resources", title="Global Resources over Time")
        st.plotly_chart(fig_res, use_container_width=True)
    with c2:
        fig_pop = px.line(df, x="Step", y="Total Population", title="Global Population over Time", color_discrete_sequence=['green'])
        st.plotly_chart(fig_pop, use_container_width=True)
    with c3:
        fig_col = px.line(df, x="Step", y="Collapses", title="Collapse Events", color_discrete_sequence=['red'])
        st.plotly_chart(fig_col, use_container_width=True)
else:
    st.info("Waiting for data...")
st.markdown("---")

# Region Cards
st.header("🌐 Region Overview")
cols = st.columns(num_regions)
for r in range(num_regions):
    res = info["resources"][r]
    ideo = info["ideologies"][r]
    last_action_idx = info.get("region_actions", [1]*num_regions)[r]
    last_action_str = ACTION_MAP.get(last_action_idx, "Unknown")
    last_reward = info.get("region_rewards", [0]*num_regions)[r]
    
    ideo = np.array(ideo)
    dom_idx = int(np.argmax(ideo))
    color_map = {0: "blue", 1: "red", 2: "green"}
    border_color = color_map.get(dom_idx, "gray")
    
    with cols[r]:
        st.markdown(f"### Region {r}")
        st.markdown(f"<div style='border-top: 5px solid {border_color}; padding-top: 10px;'></div>", unsafe_allow_html=True)
        
        # Show RL Reinforcement Info
        st.info(f"**Action:** {last_action_str} | **Reward:** {last_reward:+.2f}")
        
        st.write(f"**Population:** {res[4]:.1f}")
        st.write(f"**Stability:** {res[5]:.2f}")
        st.write(f"**Tech Level:** {res[6]:.2f}")
        
        st.write("**Resources:**")
        value = float(res[0] / res[3]) if res[3] > 0 else 0.0
        value = min(1.0, value)

        st.progress(value, text=f"Water: {float(res[0]):.0f}")
        st.progress(min(1.0, res[1]/res[3] if res[3]>0 else 0), text=f"Food: {res[1]:.0f}")
        st.progress(min(1.0, res[2]/res[3] if res[3]>0 else 0), text=f"Energy: {res[2]:.0f}")
        
        st.write("**Ideology:**")
        st.plotly_chart(plot_radar(ideo), use_container_width=True)
st.markdown("---")

# Trade Network Visualization
st.header("🤝 Trade Network")
trust_net = info.get("trust_network", {})

if trust_net:
    G = nx.DiGraph()
    for u in trust_net:
        for v in trust_net[u]:
            t = trust_net[u][v].get("trust", 0)
            if t > 0.3:  
                G.add_edge(int(u), int(v), weight=t)
                
    if len(G.edges) > 0:
        pos = nx.spring_layout(G, seed=42) # fixed seed to prevent jitter
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Region {node}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                size=30,
                color='lightblue',
                line_width=2))
                
        fig_net = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        col_net, _ = st.columns([2, 1])
        with col_net:
            st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("No active trade routes (all trust levels below 0.3).")
else:
    st.info("No trade network data found.")


# --- Live Loop ---
if st.session_state.simulation_running:
    # 1. Predict Action
    action, _states = st.session_state.model.predict(st.session_state.obs)
    
    # 2. Step Environment
    obs, reward, done, truncated, info = st.session_state.env.step(action)
    st.session_state.obs = obs
    st.session_state.latest_info = info
    st.session_state.step_count += 1
    
    # 3. Save History for UI
    total_res = sum([sum(info["resources"][r][:3]) for r in range(st.session_state.env.num_regions)])
    total_pop = sum([info["resources"][r][4] for r in range(st.session_state.env.num_regions)])
    
    st.session_state.history.append({
        "Step": st.session_state.step_count,
        "Total Resources": total_res,
        "Total Population": total_pop,
        "Collapses": info.get("global_collapses", 0)
    })
    
    # Keep history bounded so dashboard doesn't lag indefinitely
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]
    
    # 4. Handle End of Episode
    if done or truncated:
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.history = []
        
    # 5. Sleep and Rerun
    time.sleep(st.session_state.simulation_speed)
    st.rerun()
