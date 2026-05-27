"""
Streamlit demo for the Deep Dyna-Q dialog system on CamRest676.

Run:
    source venv/bin/activate
    streamlit run demo_app.py
"""
import sys, os, io, json, copy, random, contextlib, pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── project root on sys.path ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from deep_dialog import dialog_config
from deep_dialog.agents import AgentDQN
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, RuleSimulator

DATA_DIR = os.path.join(ROOT, "deep_dialog", "data", "camrest676")
CKPT_DIR = os.path.join(ROOT, "deep_dialog", "checkpoints")

# ─────────────────────────────── helpers ──────────────────────────────────

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


@st.cache_resource
def load_data():
    all_goals = _load_pickle(os.path.join(DATA_DIR, "camrest_user_goals.p"))
    kb        = _load_pickle(os.path.join(DATA_DIR, "camrest_kb.p"))
    slot_dict = _load_pickle(os.path.join(DATA_DIR, "camrest_dict.p"))
    act_set   = text_to_dict(os.path.join(DATA_DIR, "dia_acts_camrest.txt"))
    slot_set  = text_to_dict(os.path.join(DATA_DIR, "slot_set_camrest.txt"))
    goal_set  = {"train": [], "valid": [], "test": [], "all": []}
    for i, g in enumerate(all_goals):
        (goal_set["test"] if i % 5 == 1 else goal_set["train"]).append(g)
        goal_set["all"].append(g)
    return kb, slot_dict, act_set, slot_set, goal_set


def list_checkpoints():
    ckpts = {}
    for dirpath, _, files in os.walk(CKPT_DIR):
        for fname in sorted(files):
            if fname.endswith(".pkl") and "agt_9" in fname and "performance" not in fname:
                full  = os.path.join(dirpath, fname)
                label = os.path.relpath(full, CKPT_DIR)
                ckpts[label] = full
    return ckpts


def silence(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def build_dialog_manager(ckpt_path, k, kb, slot_dict, act_set, slot_set, goal_set):
    dialog_config.run_mode = 0   # NL mode so we get readable text

    agent_params = dict(
        max_turn=20, epsilon=0.0, agent_run_mode=0, agent_act_level=0,
        experience_replay_pool_size=100, dqn_hidden_size=256, batch_size=64,
        gamma=0.99, predict_mode=True, trained_model_path=ckpt_path,
        warm_start=0, cmd_input_mode=0, world_model_weight=0.5,
        per_alpha=0.6, per_beta=0.4,
    )
    agent = AgentDQN(kb, act_set, slot_set, agent_params)

    usersim_params = dict(
        max_turn=20, slot_err_probability=0.0, slot_err_mode=0,
        intent_err_probability=0.0, simulator_run_mode=0, simulator_act_level=0,
        learning_phase="all", hidden_size=256,
    )
    user_sim    = RuleSimulator(slot_dict, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dict, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)

    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(
        os.path.join(DATA_DIR, "dia_act_nl_pairs_camrest.json")
    )
    nlu_model = nlu()
    for obj in (agent, user_sim, world_model):
        obj.set_nlg_model(nlg_model)
        obj.set_nlu_model(nlu_model)

    dm = DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)
    agent.planning_steps = k
    return dm


def compute_q_values(dm):
    import torch
    state  = dm.state_tracker.get_state_for_agent()
    rep    = dm.agent.prepare_state_representation(state)
    device = next(dm.agent.dqn.parameters()).device
    dm.agent.dqn.eval()
    with torch.no_grad():
        q = dm.agent.dqn(torch.FloatTensor(rep).to(device)).cpu().numpy().flatten()
    return q


def action_label(idx):
    a = dialog_config.feasible_actions[idx]
    if a["inform_slots"]:
        return f"inform({list(a['inform_slots'].keys())[0]})"
    if a["request_slots"]:
        return f"request({list(a['request_slots'].keys())[0]})"
    return a["diaact"]


def nl_or_fallback(action_dict):
    nl = action_dict.get("nl", "").strip()
    if nl:
        return nl
    diaact = action_dict.get("diaact", "?")
    inf    = action_dict.get("inform_slots", {})
    req    = action_dict.get("request_slots", {})
    if inf:
        return f"[{diaact}: {', '.join(f'{k}={v}' for k, v in inf.items())}]"
    if req:
        return f"[{diaact}: {', '.join(req.keys())}?]"
    return f"[{diaact}]"


def q_value_chart(q_values):
    n      = len(q_values)
    labels = [action_label(i) for i in range(n)]
    order  = np.argsort(q_values)[::-1]
    top_k  = min(12, n)
    top_i  = order[:top_k]
    colors = ["#27ae60" if i == order[0] else "#2980b9" for i in top_i]
    fig = go.Figure(go.Bar(
        x=[float(q_values[i]) for i in top_i],
        y=[labels[i] for i in top_i],
        orientation="h",
        marker_color=colors,
        text=[f"{q_values[i]:.2f}" for i in top_i],
        textposition="auto",
    ))
    fig.update_layout(
        title="Agent Q-values (green = chosen action)",
        xaxis_title="Q-value",
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    return fig


def smooth(y, w=15):
    y = np.array(y, dtype=float)
    w = max(1, min(w, len(y)))
    return np.convolve(y, np.ones(w) / w, mode="same").tolist()


# ─────────────────────────────── page setup ───────────────────────────────

st.set_page_config(
    page_title="Deep Dyna-Q · CamRest676",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️  Deep Dyna-Q Dialog System — CamRest676")
st.caption(
    "Restaurant booking dialog agent — Dueling DQN + Prioritized Replay + World Model (k planning steps)"
)

# ─────────────────────────────── sidebar ──────────────────────────────────

with st.sidebar:
    st.header("⚙️ Model Settings")

    ckpts = list_checkpoints()
    if not ckpts:
        st.error("No checkpoints found under deep_dialog/checkpoints/")
        st.stop()

    default_key = next(
        (k for k in ckpts if "improved" in k and "115" in k),
        list(ckpts.keys())[0],
    )
    ckpt_label = st.selectbox(
        "Checkpoint", list(ckpts.keys()),
        index=list(ckpts.keys()).index(default_key),
    )
    k_steps = st.slider("Planning steps  k", 0, 15, 7)

    load_btn = st.button("Load Model", type="primary", use_container_width=True)
    if load_btn:
        with st.spinner("Loading dialog system…"):
            kb, slot_dict, act_set, slot_set, goal_set = load_data()
            dm = silence(
                build_dialog_manager,
                ckpts[ckpt_label], k_steps,
                kb, slot_dict, act_set, slot_set, goal_set,
            )
            st.session_state.update(
                dm=dm,
                conversation=[],
                episode_over=True,
                dialog_status=dialog_config.NO_OUTCOME_YET,
                q_values=None,
                total_reward=0.0,
                user_goal=None,
            )
        st.success("Model loaded!")

    st.divider()
    st.markdown(
        "**Best checkpoint:** `improved/agt_9_115`  \n"
        "**Success rate (200 eval):** 61 %  \n"
        "**Improvements:** Dueling DQN · PER · Double-DQN · Soft τ-update · γ=0.99"
    )

# ─────────────────────────────── tabs ─────────────────────────────────────

tab_live, tab_curves, tab_batch = st.tabs(
    ["🗨️  Live Dialog", "📈  Learning Curves", "📊  Batch Eval"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Dialog
# ══════════════════════════════════════════════════════════════════════════

with tab_live:
    if "dm" not in st.session_state:
        st.info("👈  Load a model from the sidebar to start.")
    else:
        dm      = st.session_state["dm"]
        ep_over = st.session_state.get("episode_over", True)

        col_chat, col_info = st.columns([3, 2], gap="medium")

        # ── right panel ───────────────────────────────────────────────
        with col_info:
            st.subheader("Controls")
            c1, c2 = st.columns(2)
            new_ep_btn    = c1.button("🆕  New Dialog",  type="primary", use_container_width=True)
            run_end_btn   = c2.button("⏭  Run to End",  use_container_width=True, disabled=ep_over)
            next_turn_btn = st.button("▶  Next Turn", use_container_width=True, disabled=ep_over)

            st.divider()

            # User goal card
            goal = st.session_state.get("user_goal")
            if goal:
                with st.expander("🎯  User Goal", expanded=True):
                    if goal.get("inform_slots"):
                        st.markdown("**Looking for a restaurant with:**")
                        for slot, val in goal["inform_slots"].items():
                            st.markdown(f"&nbsp;&nbsp;• **{slot}** = `{val}`")
                    if goal.get("request_slots"):
                        st.markdown("**Wants to know:**")
                        for slot in goal["request_slots"]:
                            st.markdown(f"&nbsp;&nbsp;• {slot}")

            # Episode outcome
            d_status = st.session_state.get("dialog_status", dialog_config.NO_OUTCOME_YET)
            reward   = st.session_state.get("total_reward", 0.0)
            if ep_over and d_status == dialog_config.SUCCESS_DIALOG:
                st.success(f"✅  **Dialog succeeded** — reward {reward:+.0f}")
            elif ep_over and d_status == dialog_config.FAILED_DIALOG:
                st.error(f"❌  **Dialog failed** — reward {reward:+.0f}")
            elif not ep_over:
                st.info(f"⏳  In progress — cumulative reward {reward:+.0f}")

            # Q-value chart
            q_vals = st.session_state.get("q_values")
            if q_vals is not None:
                st.plotly_chart(q_value_chart(q_vals), use_container_width=True)

        # ── left panel: conversation ───────────────────────────────────
        with col_chat:
            st.subheader("Conversation")

            # ── helper: execute one agent+user turn ──────────────────
            def do_next_turn():
                dm = st.session_state["dm"]
                with contextlib.redirect_stdout(io.StringIO()):
                    episode_over, reward = dm.next_turn(
                        record_training_data=False,
                        record_training_data_for_user=False,
                    )
                # Agent utterance
                agt_act = dm.agent_action["act_slot_response"]
                agt_nl  = nl_or_fallback(agt_act)
                turn_no = agt_act.get("turn", "")
                st.session_state["conversation"].append(
                    ("agent", f"**Turn {turn_no}:** {agt_nl}", agt_act)
                )
                # User response (always available, even on final turn)
                usr_act = dm.user_action
                usr_nl  = nl_or_fallback(usr_act)
                usr_t   = usr_act.get("turn", "")
                st.session_state["conversation"].append(
                    ("user", f"**Turn {usr_t}:** {usr_nl}", usr_act)
                )
                st.session_state["total_reward"] += reward
                st.session_state["episode_over"] = episode_over
                if episode_over:
                    st.session_state["dialog_status"] = getattr(
                        dm.user, "dialog_status", dialog_config.NO_OUTCOME_YET
                    )
                    st.session_state["q_values"] = None
                else:
                    st.session_state["q_values"] = compute_q_values(dm)

            # ── button handlers ───────────────────────────────────────
            if new_ep_btn:
                with contextlib.redirect_stdout(io.StringIO()):
                    dm.initialize_episode(use_environment=True)
                first_act = dm.user_action
                first_nl  = nl_or_fallback(first_act)
                first_t   = first_act.get("turn", 0)
                st.session_state.update(
                    conversation=[("user", f"**Turn {first_t}:** {first_nl}", first_act)],
                    episode_over=False,
                    dialog_status=dialog_config.NO_OUTCOME_YET,
                    total_reward=0.0,
                    user_goal=copy.deepcopy(dm.user.goal),
                    q_values=compute_q_values(dm),
                )
                st.rerun()

            if next_turn_btn:
                do_next_turn()
                st.rerun()

            if run_end_btn:
                while not st.session_state.get("episode_over", True):
                    do_next_turn()
                st.rerun()

            # ── render conversation ───────────────────────────────────
            conv = st.session_state.get("conversation", [])
            if not conv:
                st.markdown("*Click **🆕 New Dialog** to begin an episode.*")
            else:
                for role, text, act in conv:
                    if role == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(text)
                            with st.expander("dia-act", expanded=False):
                                st.json({
                                    "diaact":        act.get("diaact", ""),
                                    "inform_slots":  act.get("inform_slots", {}),
                                    "request_slots": act.get("request_slots", {}),
                                }, expanded=False)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(text)
                            with st.expander("dia-act", expanded=False):
                                st.json({
                                    "diaact":        act.get("diaact", ""),
                                    "inform_slots":  act.get("inform_slots", {}),
                                    "request_slots": act.get("request_slots", {}),
                                }, expanded=False)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Learning Curves
# ══════════════════════════════════════════════════════════════════════════

with tab_curves:
    st.subheader("Training Curves — Improved DQN (400 episodes)")

    perf_path = os.path.join(CKPT_DIR, "improved", "agt_9_performance_records.json")
    if not os.path.exists(perf_path):
        st.warning("Performance records not found at: " + perf_path)
    else:
        with open(perf_path) as f:
            perf = json.load(f)

        eps = sorted(perf["success_rate"].keys(), key=int)
        x   = [int(e) for e in eps]
        sr  = [perf["success_rate"][e] for e in eps]
        at  = [perf["ave_turns"][e]    for e in eps]
        ar  = [perf["ave_reward"][e]   for e in eps]
        bl  = [perf.get("bellman_loss", {}).get(e, None) for e in eps]

        # top-level metrics
        peak_ep  = x[int(np.argmax(sr))]
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Success Rate",         f"{max(sr):.3f}",
                  f"episode {peak_ep}")
        m2.metric("Final SR (mean last 20 ep)", f"{np.mean(sr[-20:]):.3f}",
                  f"± {np.std(sr[-20:]):.3f}")
        m3.metric("Final avg reward (last 20)", f"{np.mean(ar[-20:]):.2f}")

        st.divider()
        c1, c2 = st.columns(2)

        # ── success rate ──────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=sr, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#2980b9", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=smooth(sr), mode="lines", name="smoothed (w=15)",
            line=dict(color="#2980b9", width=2.5),
        ))
        fig.add_vline(x=peak_ep, line_dash="dash", line_color="orange",
                      annotation_text=f"best ckpt (ep {peak_ep})",
                      annotation_position="top right")
        fig.update_layout(title="Success Rate per Eval Window",
                          xaxis_title="Training episode", yaxis_title="Success rate",
                          yaxis_range=[-0.02, 1.02], height=300,
                          margin=dict(t=40, b=30, l=50, r=10))
        c1.plotly_chart(fig, use_container_width=True)

        # ── avg turns ────────────────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x, y=at, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#e74c3c", width=1),
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=smooth(at), mode="lines", name="smoothed",
            line=dict(color="#e74c3c", width=2.5),
        ))
        fig2.update_layout(title="Avg Turns / Dialog",
                           xaxis_title="Training episode", yaxis_title="Avg turns",
                           height=300, margin=dict(t=40, b=30, l=50, r=10))
        c2.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        # ── avg reward ───────────────────────────────────────────────
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=x, y=ar, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#27ae60", width=1),
        ))
        fig3.add_trace(go.Scatter(
            x=x, y=smooth(ar), mode="lines", name="smoothed",
            line=dict(color="#27ae60", width=2.5),
        ))
        fig3.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.8)
        fig3.update_layout(title="Avg Reward per Eval Window",
                           xaxis_title="Training episode", yaxis_title="Avg reward",
                           height=300, margin=dict(t=40, b=30, l=50, r=10))
        c3.plotly_chart(fig3, use_container_width=True)

        # ── bellman loss ──────────────────────────────────────────────
        valid_bl = [(xi, bi) for xi, bi in zip(x, bl) if bi is not None]
        if valid_bl:
            bx, by = zip(*valid_bl)
            fig4 = go.Figure(go.Scatter(
                x=list(bx), y=list(by), mode="lines", name="Bellman loss",
                line=dict(color="#9b59b6", width=1.5),
            ))
            fig4.update_layout(title="Bellman (Huber) Loss",
                               xaxis_title="Training episode", yaxis_title="Loss",
                               height=300, margin=dict(t=40, b=30, l=50, r=10))
            c4.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Eval
# ══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("Batch Evaluation")

    if "dm" not in st.session_state:
        st.info("👈  Load a model from the sidebar first.")
    else:
        n_eval = st.number_input(
            "Number of dialogs", min_value=10, max_value=500, value=100, step=10
        )

        if st.button("▶  Run Evaluation", type="primary"):
            dm        = st.session_state["dm"]
            successes = 0
            all_turns   = []
            all_rewards = []

            progress = st.progress(0.0, text="Starting…")

            for i in range(n_eval):
                with contextlib.redirect_stdout(io.StringIO()):
                    dm.initialize_episode(use_environment=True)

                ep_over    = False
                ep_reward  = 0.0
                while not ep_over:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ep_over, reward = dm.next_turn(
                            record_training_data=False,
                            record_training_data_for_user=False,
                        )
                    ep_reward += reward

                ds = getattr(dm.user, "dialog_status", dialog_config.NO_OUTCOME_YET)
                if ds == dialog_config.SUCCESS_DIALOG:
                    successes += 1
                all_turns.append(dm.user.state.get("turn", 0))
                all_rewards.append(ep_reward)

                if (i + 1) % max(1, n_eval // 50) == 0 or (i + 1) == n_eval:
                    progress.progress(
                        (i + 1) / n_eval,
                        text=f"Dialog {i + 1} / {n_eval}  —  SR so far: {successes/(i+1):.3f}",
                    )

            progress.empty()

            sr_val    = successes / n_eval
            avg_turns = float(np.mean(all_turns))
            avg_rew   = float(np.mean(all_rewards))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Success Rate",  f"{sr_val:.3f}", f"{successes}/{n_eval}")
            mc2.metric("Avg Turns",     f"{avg_turns:.1f}")
            mc3.metric("Avg Reward",    f"{avg_rew:.2f}")

            st.divider()
            ch1, ch2 = st.columns(2)

            fig_t = px.histogram(
                x=all_turns, nbins=15,
                title=f"Dialog length distribution  (n={n_eval})",
                labels={"x": "Dialog turns", "y": "Count"},
                color_discrete_sequence=["#2980b9"],
            )
            fig_t.update_layout(height=300, margin=dict(t=40, b=30))
            ch1.plotly_chart(fig_t, use_container_width=True)

            fig_r = px.histogram(
                x=all_rewards, nbins=15,
                title="Cumulative reward distribution",
                labels={"x": "Reward", "y": "Count"},
                color_discrete_sequence=["#27ae60"],
            )
            fig_r.update_layout(height=300, margin=dict(t=40, b=30))
            ch2.plotly_chart(fig_r, use_container_width=True)
