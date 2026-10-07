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


ACTION_NAMES = {0: "noop", 1: "patch", 2: "block_port", 3: "isolate", 4: "scan"}
ACTION_COLORS = {0: "#888", 1: "#2ecc71", 2: "#3498db", 3: "#e67e22", 4: "#9b59b6"}


@st.cache_resource
def load_model_and_env():
    """Load a MATCHED (model, env) pair. Fails loudly if no compatible pair exists."""
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        st.error(f"Could not import stable-baselines3: {e}")
        st.stop()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Try each version in order of preference. Model and env MUST be from the same version.
    candidates = [
        ("v7", "ppo_acds_v7"),
        ("v6", "ppo_acds_v6"),
        ("v5", "ppo_acds_v1"),
    ]

    for version, model_name in candidates:
        model_path = os.path.join(root, "models", f"{model_name}.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(root, "models", f"{model_name}_final.zip")
        if not os.path.exists(model_path):
            continue

        # Try to load the matching env for this version
        try:
            if version == "v7":
                from src.sim.environment_v7 import CyberDefenseEnvV7
                env = CyberDefenseEnvV7(n_hosts=10, max_steps=100)
            elif version == "v6":
                from src.sim.environment_v6 import CyberDefenseEnvV6
                env = CyberDefenseEnvV6(n_hosts=10, max_steps=100)
            else:
                from src.sim.environment import CyberDefenseEnv
                env = CyberDefenseEnv(n_hosts=10, max_steps=100)
        except Exception as e:
            st.warning(f"Could not import env for {version}: {e}")
            continue

        # Try to load the model
        try:
            model = PPO.load(model_path)
        except Exception as e:
            st.warning(f"Could not load model {model_name}: {e}")
            continue

        # CRITICAL: verify observation space matches
        if model.observation_space.shape != env.observation_space.shape:
            st.warning(
                f"Model/env observation mismatch for {version}: "
                f"model={model.observation_space.shape}, env={env.observation_space.shape}"
            )
            continue

        # Everything matches — return this pair
        return model, env, version

    st.error(
        "No compatible (model, env) pair found. "
        "Make sure models/ contains a trained model AND the matching "
        "environment file is committed (e.g., src/sim/environment_v7.py)."
    )
    st.stop()


def build_network_figure(env, last_target=None):
    hosts = env.net.hosts
    n = len(hosts)
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    positions = {0: (0.0, 0.0)}
    for i, h_id in enumerate([h for h in hosts.keys() if h != 0]):
        positions[h_id] = (np.cos(angles[i]), np.sin(angles[i]))

    edge_x, edge_y = [], []
    for a, b in env.net.links:
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#555"),
        hoverinfo="none", showlegend=False,
    )

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for h_id, h in hosts.items():
        x, y = positions[h_id]
        node_x.append(x); node_y.append(y)

        if h.compromised:
            color = "#e74c3c"
        elif h.health > 0.7:
            color = "#2ecc71"
        elif h.health > 0.3:
            color = "#f39c12"
        else:
            color = "#c0392b"
        node_color.append(color)

        size = 34 if h_id == last_target else 24
        node_size.append(size)

        node_text.append(
            f"<b>Host {h_id}</b> ({h.role})<br>"
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
                    line=dict(width=2, color="#222")),
        hovertext=node_text, hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=430,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    return fig


def main():
    st.set_page_config(page_title="ACDS Dashboard", page_icon="🛡️", layout="wide")
    st.title("🛡️ ACDS - Autonomous Cyber Defense Simulator")

    model, env, version = load_model_and_env()
    st.caption(
        f"Live demo: trained PPO agent ({version}) defending a 10-host network "
        f"against calibrated attacks."
    )

    with st.sidebar:
        st.header("Controls")
        episodes = st.slider("Episodes to run", 1, 10, 1)
        delay = st.slider("Step delay (seconds)", 0.0, 0.5, 0.05, 0.01)
        st.divider()
        st.markdown("**Legend**")
        st.markdown("🟢 Healthy")
        st.markdown("🟡 Damaged")
        st.markdown("🔴 Compromised")
        st.divider()
        st.markdown("**Actions**")
        for a, name in ACTION_NAMES.items():
            st.markdown(
                f"<span style='color:{ACTION_COLORS[a]}; font-weight:bold'>"
                f"{a} = {name}</span>",
                unsafe_allow_html=True,
            )

    net_ph = st.empty()
    met_ph = st.empty()
    log_ph = st.empty()

    net_ph.plotly_chart(build_network_figure(env),
                        use_container_width=True, key="init")

    run_btn = st.button("▶️ Run Episode(s)", type="primary")
    if not run_btn:
        st.info("Click **Run Episode(s)** to watch the trained agent defend the network.")
        return

    all_rewards = []
    for ep in range(episodes):
        st.subheader(f"Episode {ep + 1} / {episodes}")
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0.0
        last_action = None
        last_target = None

        for t in range(env.max_steps):
            action_batched, _ = model.predict(obs[None, :], deterministic=True)
            action = action_batched[0]

            # Branch on action space type: Discrete (v5) vs MultiDiscrete (v6/v7)
            if hasattr(model.action_space, "nvec"):
                action_type = int(action[0])
                target_host = int(action[1])
                step_action = action
            else:
                action_type = int(action)
                target_host = None
                step_action = int(action)

            target = env._most_at_risk_host()
            last_target = target.id if target is not None else None

            obs, reward, terminated, truncated, info = env.step(step_action)
            total_reward += reward
            last_action = action_type

            net_ph.plotly_chart(
                build_network_figure(env, last_target=last_target),
                use_container_width=True, key=f"fig_{ep}_{t}",
            )

            with met_ph.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Step", f"{t + 1} / 100")
                c2.metric("Reward", f"{total_reward:+.2f}")
                c3.metric("Healthy", info["n_healthy"])
                c4.metric("Compromised", info["n_compromised"])
                if last_action is not None:
                    name = ACTION_NAMES.get(last_action, "?")
                    color = ACTION_COLORS.get(last_action, "#888")
                    st.markdown(
                        f"Last action: <span style='color:{color}; "
                        f"font-weight:bold; font-size:1.1em'>{name}</span>"
                        + (f" → host {last_target}" if last_target is not None else ""),
                        unsafe_allow_html=True,
                    )

            with log_ph.container():
                events = env.net.recent_events(12)
                if events:
                    rows = [{"step": t_, "kind": k,
                             "details": ", ".join(f"{kk}={vv}" for kk, vv in p.items()
                                                  if kk != "action")}
                            for t_, k, p in events]
                    st.dataframe(rows, use_container_width=True, hide_index=True)

            if delay > 0:
                time.sleep(delay)
            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        if terminated:
            st.warning(f"Episode {ep + 1} terminated early.")
        else:
            st.success(f"Episode {ep + 1} done. Compromised: {info['n_compromised']}")

    st.divider()
    st.subheader("Summary")
    st.write(f"Mean reward over {episodes} episode(s): **{np.mean(all_rewards):+.2f}**")


if __name__ == "__main__":
    main()



