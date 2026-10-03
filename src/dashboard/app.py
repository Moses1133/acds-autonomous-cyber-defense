"""
ACDS Dashboard - live visualization of the trained PPO agent defending
the simulated network.

Run:
    streamlit run src\dashboard\app.py

Then open http://localhost:8501
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stable_baselines3 import PPO

from src.sim.environment import (
    CyberDefenseEnv,
    ACTION_NOOP, ACTION_PATCH, ACTION_BLOCK_PORT, ACTION_ISOLATE, ACTION_SCAN,
)


# ---------- constants ----------

ACTION_NAMES = {
    ACTION_NOOP:       "noop",
    ACTION_PATCH:      "patch",
    ACTION_BLOCK_PORT: "block_port",
    ACTION_ISOLATE:    "isolate",
    ACTION_SCAN:       "scan",
}
ACTION_COLORS = {
    ACTION_NOOP:       "#888",
    ACTION_PATCH:      "#2ecc71",
    ACTION_BLOCK_PORT: "#3498db",
    ACTION_ISOLATE:    "#e67e22",
    ACTION_SCAN:       "#9b59b6",
}


# ---------- helpers ----------

@st.cache_resource
def load_model():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "models", "ppo_acds_v1.zip")
    if not os.path.exists(path):
        path = os.path.join(root, "models", "ppo_acds_v1_final.zip")
    if not os.path.exists(path):
        st.error("No trained model found. Run `python scripts\\train_ppo.py` first.")
        st.stop()
    return PPO.load(path)


def build_network_figure(env, last_action=None, last_target=None):
    """Star topology: host 0 is the router, all others connect to it."""
    hosts = env.net.hosts

    # Position hosts in a circle around the router
    n = len(hosts)
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    positions = {0: (0.0, 0.0)}
    for i, h_id in enumerate([h for h in hosts.keys() if h != 0]):
        positions[h_id] = (np.cos(angles[i]), np.sin(angles[i]))

    # Edges
    edge_x, edge_y = [], []
    for a, b in env.net.links:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#bbb"),
        hoverinfo="none", showlegend=False,
    )

    # Nodes
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for h_id, h in hosts.items():
        x, y = positions[h_id]
        node_x.append(x); node_y.append(y)

        if h.compromised:
            color = "#e74c3c"           # red = compromised
        elif h.health > 0.7:
            color = "#2ecc71"           # green = healthy
        elif h.health > 0.3:
            color = "#f39c12"           # orange = damaged
        else:
            color = "#c0392b"           # dark red = critical
        node_color.append(color)

        # Border highlight if this is the last action target
        size = 32 if h_id == last_target else 24
        node_size.append(size)

        node_text.append(
            f"<b>Host {h_id}</b> ({h.role})<br>"
            f"IP: {h.ip}<br>"
            f"Health: {h.health:.2f}<br>"
            f"Ports: {h.open_ports}<br>"
            f"CVEs: {len(h.vulnerabilities)}<br>"
            f"Compromised: {h.compromised}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[f"H{i}" for i in hosts.keys()],
        textposition="middle center",
        textfont=dict(color="white", size=11),
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color="#333")),
        hovertext=node_text, hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=420,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    return fig


def render_metrics(placeholder, step, reward_total, n_healthy, n_compromised,
                   last_action, last_target):
    with placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Step", f"{step} / 100")
        c2.metric("Total Reward", f"{reward_total:+.2f}")
        c3.metric("Healthy Hosts", f"{n_healthy}")
        c4.metric("Compromised", f"{n_compromised}", delta=None)
        if last_action is not None:
            name = ACTION_NAMES.get(int(last_action), f"action {last_action}")
            color = ACTION_COLORS.get(int(last_action), "#888")
            st.markdown(
                f"Last action: <span style='color:{color}; font-weight:bold; "
                f"font-size:1.1em'>{name}</span>"
                + (f" → target host {last_target}" if last_target is not None else ""),
                unsafe_allow_html=True,
            )


def render_event_log(placeholder, events):
    with placeholder.container():
        if not events:
            st.info("No events yet.")
            return
        rows = []
        for (t, kind, payload) in events[-12:]:
            payload_s = ", ".join(f"{k}={v}" for k, v in payload.items()
                                  if k not in ("action",))
            rows.append({"step": t, "kind": kind, "details": payload_s})
        st.dataframe(rows, width='stretch', hide_index=True)


# ---------- main app ----------

def main():
    st.set_page_config(page_title="ACDS Dashboard", page_icon="🛡️", layout="wide")

    st.title("🛡️ ACDS - Autonomous Cyber Defense Simulator")
    st.caption("Live demo: trained PPO agent defending a 10-host network.")

    # Sidebar
    with st.sidebar:
        st.header("Controls")
        episodes = st.slider("Episodes to run", 1, 10, 1)
        delay = st.slider("Step delay (seconds)", 0.0, 0.5, 0.05, 0.01)
        st.divider()
        st.markdown("**Legend**")
        st.markdown("🟢 Healthy host")
        st.markdown("🟡 Damaged host")
        st.markdown("🔴 Compromised host")
        st.divider()
        st.markdown("**Actions**")
        for a, name in ACTION_NAMES.items():
            st.markdown(
                f"<span style='color:{ACTION_COLORS[a]}; font-weight:bold'>"
                f"{a} = {name}</span>",
                unsafe_allow_html=True,
            )

    model = load_model()
    env = CyberDefenseEnv(n_hosts=10, max_steps=100)

    run_btn = st.button("▶️ Run Episode(s)", type="primary")

    net_placeholder    = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder    = st.empty()

    # Initial render (before running)
    fig = build_network_figure(env)
    net_placeholder.plotly_chart(fig, width='stretch',
                                 key="initial_fig")

    if not run_btn:
        st.info("Click **Run Episode(s)** to watch the trained agent defend "
                "the network.")
        return

    all_rewards = []
    for ep in range(episodes):
        st.subheader(f"Episode {ep + 1} / {episodes}")

        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0
        last_action = None
        last_target = None

        for t in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Peek at what the action will target (before stepping)
            target = env._most_at_risk_host()
            last_target = target.id if target is not None else None

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            last_action = action

            # Update network graph
            fig = build_network_figure(env, last_action=action,
                                        last_target=last_target)
            net_placeholder.plotly_chart(fig, width='stretch',
                                         key=f"fig_{ep}_{t}")

            # Update metrics
            render_metrics(
                metrics_placeholder,
                step=t + 1, reward_total=total_reward,
                n_healthy=info["n_healthy"], n_compromised=info["n_compromised"],
                last_action=action, last_target=last_target,
            )

            # Update event log
            render_event_log(log_placeholder, env.net.recent_events(12))

            if delay > 0:
                time.sleep(delay)

            if terminated or truncated:
                break

        all_rewards.append(total_reward)

        if terminated:
            st.warning("Episode terminated early - network collapsed.")
        else:
            st.success(f"Episode {ep + 1} done. "
                       f"Compromised at end: {info['n_compromised']}")

    st.divider()
    st.subheader("Summary")
    st.write(f"Mean reward across {episodes} episode(s): "
             f"**{np.mean(all_rewards):+.2f}**")


if __name__ == "__main__":
    main()



