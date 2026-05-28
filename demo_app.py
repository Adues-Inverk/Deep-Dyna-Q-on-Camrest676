"""
Streamlit demo for the Deep Dyna-Q dialog system on CamRest676.

Run:
    streamlit run demo_app.py

New in this version
-------------------
* phi3:3.8b NLG via Ollama — sidebar toggle rewrites every utterance using
  a local LLM for natural-sounding restaurant-booking conversation.
* k-Sweep tab — overlaid training curves for k=0 … k=10 from sweep_k/ data,
  plus a side-by-side k=0 vs k=5 panel once the full training runs complete.
* Before / After RL — metrics and sample dialog from performance_records.json
  are surfaced directly in the Learning Curves tab.
"""
import sys, os, io, json, copy, random, contextlib, pickle
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from deep_dialog import dialog_config
from deep_dialog.agents import AgentDQN
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, RuleSimulator

DATA_DIR   = os.path.join(ROOT, "deep_dialog", "data", "camrest676")
CKPT_DIR   = os.path.join(ROOT, "deep_dialog", "checkpoints")
OLLAMA_URL = "http://localhost:11434"

# ─────────────────────────────── data helpers ──────────────────────────────

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
    dialog_config.run_mode = 0

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
    nl_text = action_dict.get("nl", "").strip()
    if nl_text:
        return nl_text
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


# ───────────────────────── Ollama / phi3 helpers ───────────────────────────

def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False


def phi3_generate(role: str, diaact: str, inform_slots: dict, request_slots: dict) -> str | None:
    """
    Call Ollama phi3:3.8b to rewrite a structured dialog act as natural language.
    Returns None if Ollama is unavailable or returns an empty response.
    """
    if role == "agent":
        system = (
            "You are a polite, concise restaurant-booking assistant. "
            "Given the dialog act, write EXACTLY ONE natural sentence. "
            "Do NOT include slot names literally; weave values into fluent English."
        )
        clean_inf = {k: v for k, v in inform_slots.items() if k != "taskcomplete"}
        task_done = inform_slots.get("taskcomplete") == "Match Available"

        if diaact == "request":
            slot = next(iter(request_slots), "preference")
            prompt = f"Ask the customer for their {slot} preference."
        elif diaact == "inform" and task_done:
            details = "; ".join(f"{k}: {v}" for k, v in clean_inf.items())
            prompt = f"Tell the customer you found a match and share: {details or 'the details'}."
        elif diaact == "inform":
            details = "; ".join(f"{k}: {v}" for k, v in clean_inf.items()) or "no specific details"
            prompt = f"Inform the customer: {details}."
        elif diaact == "thanks":
            prompt = "Politely close the conversation."
        else:
            prompt = f"Perform dialog act '{diaact}'."

    else:  # user / customer
        system = (
            "You are a customer looking to book a restaurant. "
            "Given the dialog act, write EXACTLY ONE natural sentence as you would say it. "
            "Be brief and natural."
        )
        if diaact == "request":
            slots = ", ".join(request_slots) or "information"
            prompt = f"Ask for: {slots}."
        elif diaact == "inform":
            details = "; ".join(f"{k}: {v}" for k, v in inform_slots.items()) or "a preference"
            prompt = f"Tell the assistant: {details}."
        elif diaact == "thanks":
            prompt = "Thank the assistant and say goodbye."
        elif diaact == "deny":
            prompt = "Politely indicate that the response did not meet your needs."
        elif diaact == "confirm_question":
            prompt = "Ask to confirm a detail."
        else:
            prompt = f"Express the dialog act '{diaact}' naturally."

    full_prompt = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n"
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "phi3:3.8b",
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 80},
            },
            timeout=15,
        )
        if r.ok:
            text = r.json().get("response", "").strip()
            # strip any system/assistant prefixes that leak through
            for tag in ("<|assistant|>", "<|end|>", "<|user|>", "<|system|>"):
                text = text.replace(tag, "").strip()
            return text or None
    except Exception:
        pass
    return None


# ─────────────────────────────── page setup ────────────────────────────────

st.set_page_config(
    page_title="Deep Dyna-Q · CamRest676",
    page_icon="🍽️",
    layout="wide",
)
st.title("🍽️  Deep Dyna-Q Dialog System — CamRest676")
st.caption(
    "Restaurant-booking agent — Dueling DQN + Prioritised Replay + World-Model planning (Dyna-Q)"
)

# ─────────────────────────────── sidebar ───────────────────────────────────

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

    # ── phi3 NLG ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 phi3:3.8b NLG (Ollama)")
    phi3_on = st.toggle(
        "Rewrite utterances with phi3",
        value=False,
        key="phi3_on",
        help="Uses Ollama phi3:3.8b to generate fluent natural language from dialog acts.",
    )
    if phi3_on:
        ollama_ok = check_ollama()
        if ollama_ok:
            st.success("Ollama reachable ✓", icon="✅")
        else:
            st.warning(
                "Ollama not responding.  \n"
                "Start it with:  \n"
                "`ollama serve`  \n"
                "then pull the model:  \n"
                "`ollama pull phi3:3.8b`",
                icon="⚠️",
            )

    st.divider()
    st.markdown(
        "**Best checkpoint:** `improved/agt_9_115`  \n"
        "**Success rate (200 eval):** 61 %  \n"
        "**Architecture:** Dueling DQN · PER · Double-DQN · Soft τ · γ=0.99  \n\n"
        "**Training:** run `python train_comparison.py` to produce  \n"
        "k=0 vs k=5 comparison checkpoints."
    )

# ─────────────────────────────── tabs ──────────────────────────────────────

tab_live, tab_curves, tab_sweep, tab_batch = st.tabs(
    ["🗨️ Live Dialog", "📈 Learning Curves", "🔬 k=0 vs k>0", "📊 Batch Eval"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Dialog
# ═══════════════════════════════════════════════════════════════════════════

with tab_live:
    if "dm" not in st.session_state:
        st.info("👈  Load a model from the sidebar to start.")
    else:
        dm      = st.session_state["dm"]
        ep_over = st.session_state.get("episode_over", True)
        phi3_active = st.session_state.get("phi3_on", False)

        col_chat, col_info = st.columns([3, 2], gap="medium")

        # ── right info panel ──────────────────────────────────────────────
        with col_info:
            st.subheader("Controls")
            c1, c2 = st.columns(2)
            new_ep_btn    = c1.button("🆕 New Dialog",  type="primary", use_container_width=True)
            run_end_btn   = c2.button("⏭ Run to End",  use_container_width=True, disabled=ep_over)
            next_turn_btn = st.button("▶ Next Turn", use_container_width=True, disabled=ep_over)

            st.divider()

            goal = st.session_state.get("user_goal")
            if goal:
                with st.expander("🎯 User Goal", expanded=True):
                    if goal.get("inform_slots"):
                        st.markdown("**Looking for a restaurant with:**")
                        for slot, val in goal["inform_slots"].items():
                            st.markdown(f"&nbsp;&nbsp;• **{slot}** = `{val}`")
                    if goal.get("request_slots"):
                        st.markdown("**Wants to know:**")
                        for slot in goal["request_slots"]:
                            st.markdown(f"&nbsp;&nbsp;• {slot}")

            d_status = st.session_state.get("dialog_status", dialog_config.NO_OUTCOME_YET)
            reward   = st.session_state.get("total_reward", 0.0)
            if ep_over and d_status == dialog_config.SUCCESS_DIALOG:
                st.success(f"✅ **Dialog succeeded** — reward {reward:+.0f}")
            elif ep_over and d_status == dialog_config.FAILED_DIALOG:
                st.error(f"❌ **Dialog failed** — reward {reward:+.0f}")
            elif not ep_over:
                st.info(f"⏳ In progress — cumulative reward {reward:+.0f}")

            q_vals = st.session_state.get("q_values")
            if q_vals is not None:
                st.plotly_chart(q_value_chart(q_vals), use_container_width=True)

        # ── conversation panel ────────────────────────────────────────────
        with col_chat:
            if phi3_active:
                st.subheader("Conversation  *(phi3 NLG active)*")
            else:
                st.subheader("Conversation")

            def _phi3_if_on(role, act):
                if not phi3_active:
                    return None
                return phi3_generate(
                    role,
                    act.get("diaact", ""),
                    act.get("inform_slots", {}),
                    act.get("request_slots", {}),
                )

            def do_next_turn():
                _dm = st.session_state["dm"]
                with contextlib.redirect_stdout(io.StringIO()):
                    episode_over, reward = _dm.next_turn(
                        record_training_data=False,
                        record_training_data_for_user=False,
                    )
                agt_act  = _dm.agent_action["act_slot_response"]
                agt_nl   = nl_or_fallback(agt_act)
                turn_no  = agt_act.get("turn", "")
                agt_phi3 = _phi3_if_on("agent", agt_act)
                st.session_state["conversation"].append(
                    ("agent", f"**Turn {turn_no}:** {agt_nl}", agt_phi3, agt_act)
                )

                usr_act  = _dm.user_action
                usr_nl   = nl_or_fallback(usr_act)
                usr_t    = usr_act.get("turn", "")
                usr_phi3 = _phi3_if_on("user", usr_act)
                st.session_state["conversation"].append(
                    ("user", f"**Turn {usr_t}:** {usr_nl}", usr_phi3, usr_act)
                )

                st.session_state["total_reward"] += reward
                st.session_state["episode_over"] = episode_over
                if episode_over:
                    st.session_state["dialog_status"] = getattr(
                        _dm.user, "dialog_status", dialog_config.NO_OUTCOME_YET
                    )
                    st.session_state["q_values"] = None
                else:
                    st.session_state["q_values"] = compute_q_values(_dm)

            # button handlers ──────────────────────────────────────────────
            if new_ep_btn:
                with contextlib.redirect_stdout(io.StringIO()):
                    dm.initialize_episode(use_environment=True)
                first_act  = dm.user_action
                first_nl   = nl_or_fallback(first_act)
                first_t    = first_act.get("turn", 0)
                first_phi3 = _phi3_if_on("user", first_act)
                st.session_state.update(
                    conversation=[("user", f"**Turn {first_t}:** {first_nl}", first_phi3, first_act)],
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

            # render conversation ──────────────────────────────────────────
            conv = st.session_state.get("conversation", [])
            if not conv:
                st.markdown("*Click **🆕 New Dialog** to begin an episode.*")
            else:
                for role, template_text, phi3_text, act in conv:
                    if role == "user":
                        with st.chat_message("user", avatar="👤"):
                            if phi3_text:
                                st.markdown(phi3_text)
                                st.caption(f"Template: {template_text}")
                            else:
                                st.markdown(template_text)
                            with st.expander("dia-act", expanded=False):
                                st.json({
                                    "diaact":        act.get("diaact", ""),
                                    "inform_slots":  act.get("inform_slots", {}),
                                    "request_slots": act.get("request_slots", {}),
                                }, expanded=False)
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            if phi3_text:
                                st.markdown(phi3_text)
                                st.caption(f"Template: {template_text}")
                            else:
                                st.markdown(template_text)
                            with st.expander("dia-act", expanded=False):
                                st.json({
                                    "diaact":        act.get("diaact", ""),
                                    "inform_slots":  act.get("inform_slots", {}),
                                    "request_slots": act.get("request_slots", {}),
                                }, expanded=False)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Learning Curves
# ═══════════════════════════════════════════════════════════════════════════

with tab_curves:
    st.subheader("Training Curves — Improved DQN")

    perf_path = os.path.join(CKPT_DIR, "improved", "agt_9_performance_records.json")
    if not os.path.exists(perf_path):
        st.warning("Performance records not found at: " + perf_path)
    else:
        with open(perf_path, encoding="utf-8") as f:
            perf = json.load(f)

        eps = sorted(perf["success_rate"].keys(), key=int)
        x   = [int(e) for e in eps]
        sr  = [perf["success_rate"][e] for e in eps]
        at  = [perf["ave_turns"][e]    for e in eps]
        ar  = [perf["ave_reward"][e]   for e in eps]
        bl  = [perf.get("bellman_loss", {}).get(e, None) for e in eps]

        peak_ep = x[int(np.argmax(sr))]
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Success Rate",          f"{max(sr):.3f}", f"episode {peak_ep}")
        m2.metric("Final SR (mean last 20 ep)", f"{np.mean(sr[-20:]):.3f}",
                  f"± {np.std(sr[-20:]):.3f}")
        m3.metric("Final avg reward (last 20)", f"{np.mean(ar[-20:]):.2f}")

        # ── Before / After RL ─────────────────────────────────────────────
        mab = perf.get("metrics_ab")
        if mab:
            st.divider()
            st.subheader("Before vs. After RL Training")
            bm = mab.get("before_rl", {})
            am = mab.get("after_best_policy", {})
            n_eval = mab.get("eval_episodes", "?")
            st.caption(f"Evaluated over {n_eval} dialogs each · best policy from epoch {am.get('best_rl_epoch', '?')}")
            ba1, ba2, ba3 = st.columns(3)
            for col, key, label, fmt in [
                (ba1, "success_rate", "Success Rate",  ".3f"),
                (ba2, "ave_turns",    "Avg Turns",     ".1f"),
                (ba3, "ave_reward",   "Avg Reward",    ".2f"),
            ]:
                bv = bm.get(key, 0)
                av = am.get(key, 0)
                col.metric(label, f"{av:{fmt}}", f"{av - bv:+{fmt}} vs before RL")

        st.divider()
        c1, c2 = st.columns(2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=sr, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#2980b9", width=1)))
        fig.add_trace(go.Scatter(x=x, y=smooth(sr), mode="lines", name="smoothed (w=15)",
            line=dict(color="#2980b9", width=2.5)))
        fig.add_vline(x=peak_ep, line_dash="dash", line_color="orange",
                      annotation_text=f"best ckpt (ep {peak_ep})",
                      annotation_position="top right")
        fig.update_layout(title="Success Rate per Eval Window",
                          xaxis_title="Training episode", yaxis_title="Success rate",
                          yaxis_range=[-0.02, 1.02], height=300,
                          margin=dict(t=40, b=30, l=50, r=10))
        c1.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=at, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#e74c3c", width=1)))
        fig2.add_trace(go.Scatter(x=x, y=smooth(at), mode="lines", name="smoothed",
            line=dict(color="#e74c3c", width=2.5)))
        fig2.update_layout(title="Avg Turns / Dialog",
                           xaxis_title="Training episode", yaxis_title="Avg turns",
                           height=300, margin=dict(t=40, b=30, l=50, r=10))
        c2.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x, y=ar, mode="lines", name="raw",
            opacity=0.2, line=dict(color="#27ae60", width=1)))
        fig3.add_trace(go.Scatter(x=x, y=smooth(ar), mode="lines", name="smoothed",
            line=dict(color="#27ae60", width=2.5)))
        fig3.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.8)
        fig3.update_layout(title="Avg Reward per Eval Window",
                           xaxis_title="Training episode", yaxis_title="Avg reward",
                           height=300, margin=dict(t=40, b=30, l=50, r=10))
        c3.plotly_chart(fig3, use_container_width=True)

        valid_bl = [(xi, bi) for xi, bi in zip(x, bl) if bi is not None]
        if valid_bl:
            bx, by = zip(*valid_bl)
            fig4 = go.Figure(go.Scatter(x=list(bx), y=list(by), mode="lines",
                name="Bellman loss", line=dict(color="#9b59b6", width=1.5)))
            fig4.update_layout(title="Bellman (Huber) Loss",
                               xaxis_title="Training episode", yaxis_title="Loss",
                               height=300, margin=dict(t=40, b=30, l=50, r=10))
            c4.plotly_chart(fig4, use_container_width=True)

        # ── Sample dialog before / after ─────────────────────────────────
        sad = perf.get("sample_dialog_ab")
        if sad:
            st.divider()
            st.subheader("Sample Dialog — Before vs. After RL")
            goal = sad.get("goal", {})
            b_out = sad.get("before_outcome", {})
            a_out = sad.get("after_outcome", {})
            st.caption(
                f"Fixed RNG seed {sad.get('rng_seed')} · "
                f"goal: {goal.get('inform_slots',{})} · "
                f"wants: {list(goal.get('request_slots',{}).keys())}"
            )
            dc1, dc2 = st.columns(2)
            with dc1:
                label_b = f"Before RL  (reward {b_out.get('reward')}, {b_out.get('turn_count')} turns)"
                st.markdown(f"**{label_b}**")
                st.code(sad.get("before_rl_conversation", ""), language=None)
            with dc2:
                label_a = (f"After RL — epoch {sad.get('best_rl_epoch')}  "
                           f"(reward {a_out.get('reward')}, {a_out.get('turn_count')} turns)")
                st.markdown(f"**{label_a}**")
                st.code(sad.get("after_best_conversation", ""), language=None)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — k=0 vs k>0 Sweep
# ═══════════════════════════════════════════════════════════════════════════

with tab_sweep:
    st.subheader("Planning Steps k — k=0 (no planning) vs. k>0 (Dyna-Q)")
    st.caption(
        "**k=0** is the pure-DQN baseline (dotted red). "
        "**k>0** adds Dyna-Q world-model planning rollouts per real episode — "
        "the agent learns faster and achieves higher success rates."
    )

    sweep_base = os.path.join(CKPT_DIR, "sweep_k")
    k_dirs = sorted(
        [d for d in os.listdir(sweep_base) if d.startswith("k_")],
        key=lambda s: int(s.split("_")[1]),
    ) if os.path.isdir(sweep_base) else []

    SWEEP_COLORS = {
        "k_0":  "#e74c3c",   # red  — baseline (dotted)
        "k_1":  "#e67e22",
        "k_3":  "#f1c40f",
        "k_5":  "#2ecc71",
        "k_7":  "#1abc9c",
        "k_10": "#3498db",
    }

    k_data: dict = {}
    for kd in k_dirs:
        rec_p = os.path.join(sweep_base, kd, "agt_9_performance_records.json")
        if os.path.exists(rec_p):
            with open(rec_p, encoding="utf-8") as f:
                k_data[kd] = json.load(f)

    if not k_data:
        st.warning("No sweep_k performance records found.  Run the training sweep first.")
    else:
        # ── overlay success-rate ──────────────────────────────────────────
        fig_sr = go.Figure()
        summary_rows = []
        for kd in sorted(k_data, key=lambda s: int(s.split("_")[1])):
            rec   = k_data[kd]
            k_val = int(kd.split("_")[1])
            eps_k = sorted(rec["success_rate"].keys(), key=int)
            x_k   = [int(e) for e in eps_k]
            sr_k  = [rec["success_rate"][e] for e in eps_k]
            color = SWEEP_COLORS.get(kd, "#95a5a6")
            is_k0 = k_val == 0

            fig_sr.add_trace(go.Scatter(
                x=x_k, y=sr_k,
                mode="lines",
                name=f"k={k_val}  (peak {max(sr_k):.2f})",
                line=dict(color=color, width=2.5 if not is_k0 else 2,
                          dash="dot" if is_k0 else "solid"),
                opacity=0.9,
            ))

            mab = rec.get("metrics_ab", {})
            ab_after = mab.get("after_best_policy", {})
            summary_rows.append({
                "k": k_val,
                "k=0 baseline?": "✓" if is_k0 else "—",
                "Peak SR": round(max(sr_k), 3),
                "Peak epoch": x_k[int(np.argmax(sr_k))],
                "After-RL SR": round(ab_after.get("success_rate", 0), 3) if ab_after else "—",
                "After-RL avg turns": round(ab_after.get("ave_turns", 0), 1) if ab_after else "—",
                "ΔSR vs before": (f"{mab.get('delta',{}).get('success_rate', 0):+.3f}"
                                  if mab else "—"),
            })

        # Shade the k=0 region to emphasise the gap
        k0_peak = max(k_data["k_0"]["success_rate"].values()) if "k_0" in k_data else 0
        fig_sr.add_hline(
            y=k0_peak, line_dash="dash", line_color="#e74c3c", line_width=1,
            annotation_text=f"k=0 peak ({k0_peak:.2f})", annotation_position="right",
        )

        fig_sr.update_layout(
            title="Success Rate vs Training Episodes — All k Values  (dotted = k=0 baseline)",
            xaxis_title="Training episode",
            yaxis_title="Success rate (eval window, 50 dialogs)",
            yaxis_range=[-0.02, 1.05],
            height=430,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(t=50, b=40, l=60, r=180),
        )
        st.plotly_chart(fig_sr, use_container_width=True)

        # ── reward overlay ────────────────────────────────────────────────
        fig_rw = go.Figure()
        for kd in sorted(k_data, key=lambda s: int(s.split("_")[1])):
            k_val = int(kd.split("_")[1])
            eps_k = sorted(k_data[kd]["ave_reward"].keys(), key=int)
            x_k   = [int(e) for e in eps_k]
            ar_k  = [k_data[kd]["ave_reward"][e] for e in eps_k]
            color = SWEEP_COLORS.get(kd, "#95a5a6")
            fig_rw.add_trace(go.Scatter(
                x=x_k, y=ar_k, mode="lines", name=f"k={k_val}",
                line=dict(color=color, width=2 if k_val > 0 else 1.5,
                          dash="dot" if k_val == 0 else "solid"),
                opacity=0.85,
            ))
        fig_rw.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.8)
        fig_rw.update_layout(
            title="Avg Reward — All k Values  (positive = net success)",
            xaxis_title="Training episode", yaxis_title="Avg reward",
            height=300,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(t=40, b=30, l=60, r=180),
        )
        st.plotly_chart(fig_rw, use_container_width=True)

        # ── summary table ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Summary — 50-episode sweep (sweep_k/)")
        summary_rows.sort(key=lambda r: r["k"])
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        # ── headline comparison ───────────────────────────────────────────
        if "k_0" in k_data and len(k_data) > 1:
            best_kd   = max(
                (kd for kd in k_data if kd != "k_0"),
                key=lambda kd: max(k_data[kd]["success_rate"].values()),
            )
            best_k    = int(best_kd.split("_")[1])
            k0_sr_pk  = max(k_data["k_0"]["success_rate"].values())
            kn_sr_pk  = max(k_data[best_kd]["success_rate"].values())

            st.divider()
            st.subheader("k=0 (No Planning) vs. Best k — Headline Numbers")
            hl1, hl2, hl3 = st.columns(3)
            hl1.metric("k=0 — peak SR", f"{k0_sr_pk:.3f}", help="Pure DQN baseline, no world model")
            hl2.metric(f"k={best_k} — peak SR", f"{kn_sr_pk:.3f}",
                       delta=f"{kn_sr_pk - k0_sr_pk:+.3f} vs k=0")
            # avg-turns comparison from metrics_ab
            k0_at  = k_data["k_0"].get("metrics_ab", {}).get("after_best_policy", {}).get("ave_turns")
            kn_at  = k_data[best_kd].get("metrics_ab", {}).get("after_best_policy", {}).get("ave_turns")
            if k0_at and kn_at:
                hl3.metric(f"Avg turns  k=0 → k={best_k}",
                           f"{k0_at:.1f} → {kn_at:.1f}",
                           delta=f"{kn_at - k0_at:+.1f}")

        # ── full 300-ep runs (populated after train_comparison.py) ────────
        best_dir = os.path.join(CKPT_DIR, "best")
        k0_rec  = os.path.join(best_dir, "k0", "agt_9_performance_records.json")
        k5_rec  = os.path.join(best_dir, "k5", "agt_9_performance_records.json")
        have_k0 = os.path.exists(k0_rec)
        have_k5 = os.path.exists(k5_rec)

        if have_k0 or have_k5:
            st.divider()
            st.subheader("Full 300-Episode Training Runs — best/k0 vs best/k5")
            st.caption("Generated by `python train_comparison.py`.  Longer runs → cleaner curves.")

            fig_full = go.Figure()
            for label, path, color, dash in [
                ("k=0  (no planning)", k0_rec, "#e74c3c", "dot"),
                ("k=5  (Dyna-Q)",      k5_rec, "#2ecc71", "solid"),
            ]:
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                eps_b = sorted(rec["success_rate"].keys(), key=int)
                x_b   = [int(e) for e in eps_b]
                sr_b  = [rec["success_rate"][e] for e in eps_b]
                fig_full.add_trace(go.Scatter(
                    x=x_b, y=sr_b, mode="lines", name=f"{label} raw",
                    opacity=0.2, line=dict(color=color, width=1, dash=dash)))
                fig_full.add_trace(go.Scatter(
                    x=x_b, y=smooth(sr_b, 20), mode="lines", name=f"{label} smoothed",
                    line=dict(color=color, width=2.5, dash=dash)))

            fig_full.update_layout(
                title="k=0 vs k=5 — Success Rate (300 episodes, smoothed w=20)",
                xaxis_title="Training episode", yaxis_title="Success rate",
                yaxis_range=[-0.02, 1.05], height=380,
                margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # before/after RL for both
            full_rows = []
            for label, path, k_v in [("k=0", k0_rec, 0), ("k=5", k5_rec, 5)]:
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                mab = rec.get("metrics_ab", {})
                if mab:
                    full_rows.append({
                        "Run": label,
                        "Before SR": mab.get("before_rl", {}).get("success_rate", "—"),
                        "After SR":  mab.get("after_best_policy", {}).get("success_rate", "—"),
                        "ΔSR":       f"{mab.get('delta',{}).get('success_rate', 0):+.3f}",
                        "After avg turns": mab.get("after_best_policy", {}).get("ave_turns", "—"),
                        "After avg reward": mab.get("after_best_policy", {}).get("ave_reward", "—"),
                    })
            if full_rows:
                st.dataframe(full_rows, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Full 300-episode comparison runs not yet available.  \n"
                "Run:  `python train_comparison.py`  \n"
                "k=0 trains first (~10 min), then k=5 (~20 min).  \n"
                "Curves will appear here automatically once the JSON files are saved.",
                icon="ℹ️",
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Batch Eval
# ═══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("Batch Evaluation")

    if "dm" not in st.session_state:
        st.info("👈  Load a model from the sidebar first.")
    else:
        n_eval = st.number_input(
            "Number of dialogs", min_value=10, max_value=500, value=100, step=10
        )

        if st.button("▶ Run Evaluation", type="primary"):
            dm        = st.session_state["dm"]
            successes = 0
            all_turns   = []
            all_rewards = []
            progress = st.progress(0.0, text="Starting…")

            for i in range(n_eval):
                with contextlib.redirect_stdout(io.StringIO()):
                    dm.initialize_episode(use_environment=True)
                ep_over   = False
                ep_reward = 0.0
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
            mc1.metric("Success Rate", f"{sr_val:.3f}", f"{successes}/{n_eval}")
            mc2.metric("Avg Turns",    f"{avg_turns:.1f}")
            mc3.metric("Avg Reward",   f"{avg_rew:.2f}")

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
