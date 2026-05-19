"""
Load the best DDQ checkpoint, roll out many dialogs in predict mode,
pick the one that BEST demonstrates the trained policy (short + successful),
and save the transcript as an English text file and a matplotlib figure.

Output:
    slides_figures/best_dialog.txt
    slides_figures/best_dialog.md
    slides_figures/best_dialog.png

Run:
    python pick_best_dialog.py
"""

import copy
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)) or "./")

from deep_dialog import dialog_config
from deep_dialog.agents import AgentDQN
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.nlg import nlg
from deep_dialog.nlu import nlu
from deep_dialog.usersims import ModelBasedSimulator, RuleSimulator


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "deep_dialog", "data", "camrest676")
CHK = os.path.join(HERE, "deep_dialog", "checkpoints")
OUT = os.path.join(HERE, "slides_figures")
os.makedirs(OUT, exist_ok=True)


# ----- pick a checkpoint -----------------------------------------------------
# Choose the file with the highest success rate suffix.
def pick_best_checkpoint():
    candidates = []
    for fn in os.listdir(CHK):
        if not fn.startswith("agt_9_") or not fn.endswith(".pkl"):
            continue
        try:
            rate = float(fn.rsplit("_", 1)[-1].replace(".pkl", ""))
            candidates.append((rate, fn))
        except ValueError:
            continue
    if not candidates:
        raise SystemExit("no agt_9_*.pkl checkpoint in %s" % CHK)
    candidates.sort(reverse=True)
    print("checkpoint candidates (top 5):")
    for r, f in candidates[:5]:
        print("  %.4f  %s" % (r, f))
    return os.path.join(CHK, candidates[0][1])


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_dialog_manager(max_turn=30):
    goal_set_all = load_pickle(os.path.join(DATA, "camrest_user_goals.p"))
    kb = load_pickle(os.path.join(DATA, "camrest_kb.p"))
    slot_dictionary = load_pickle(os.path.join(DATA, "camrest_dict.p"))
    act_set = text_to_dict(os.path.join(DATA, "dia_acts_camrest.txt"))
    slot_set = text_to_dict(os.path.join(DATA, "slot_set_camrest.txt"))

    split_fold = 5
    goal_set = {"train": [], "valid": [], "test": [], "all": []}
    for i, g in enumerate(goal_set_all):
        (goal_set["test"] if i % split_fold == 1 else goal_set["train"]).append(g)
        goal_set["all"].append(g)
    dialog_config.run_mode = 1

    agent_params = {
        "max_turn": max_turn, "epsilon": 0.0, "agent_run_mode": 1, "agent_act_level": 0,
        "experience_replay_pool_size": 5000, "dqn_hidden_size": 60, "batch_size": 16,
        "gamma": 0.9, "predict_mode": True, "trained_model_path": None,
        "warm_start": 0, "cmd_input_mode": 0,
    }
    agent = AgentDQN(kb, act_set, slot_set, agent_params)

    usersim_params = {
        "max_turn": max_turn, "slot_err_probability": 0.0, "slot_err_mode": 0,
        "intent_err_probability": 0.0, "simulator_run_mode": 1,
        "simulator_act_level": 0, "learning_phase": "all", "hidden_size": 60,
    }
    user_sim = RuleSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(slot_dictionary, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)

    nlg_model = nlg()
    nlg_model.load_predefine_act_nl_pairs(os.path.join(DATA, "dia_act_nl_pairs_camrest.json"))
    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)
    world_model.set_nlg_model(nlg_model)

    nlu_model = nlu()
    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)
    world_model.set_nlu_model(nlu_model)

    dm = DialogManager(agent, user_sim, world_model, act_set, slot_set, kb)
    return dm, agent


def summarize_goal_en(goal):
    parts = []
    for slot, val in (goal.get("inform_slots") or {}).items():
        parts.append("%s = %s" % (slot, val))
    req = list((goal.get("request_slots") or {}).keys())
    if req:
        parts.append("wants to know: " + ", ".join(req))
    return "; ".join(parts) if parts else "(empty goal)"


def format_dialog_en(goal, history, outcome, nlg_model):
    lines = []
    lines.append("User goal: " + summarize_goal_en(goal))
    lines.append("")
    lines.append("------------- Conversation -------------")
    for h in history or []:
        speaker = h.get("speaker", "")
        role = "System " if speaker == "agent" else "User   "
        turn_msg = "agt" if speaker == "agent" else "usr"
        dia_act = {
            "diaact": h.get("diaact"),
            "inform_slots": copy.deepcopy(h.get("inform_slots") or {}),
            "request_slots": copy.deepcopy(h.get("request_slots") or {}),
        }
        utt = nlg_model.convert_diaact_to_nl(dia_act, turn_msg)
        lines.append("%s: %s" % (role, utt))
    lines.append("----------------------------------------")
    lines.append("")
    lines.append("Result: %s | reward = %s | turns = %s" % (
        "SUCCESS" if outcome["success"] else "FAIL",
        outcome["reward"], outcome["turn_count"],
    ))
    return "\n".join(lines)


def rollout(dm, agent, rng_seed, torch_seed=100):
    """One predict-mode rollout with a given RNG."""
    numpy.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.manual_seed(torch_seed)
    agent.predict_mode = True
    dm.initialize_episode(True)
    goal = copy.deepcopy(dm.user.goal)
    episode_over = False
    reward = 0.0
    while not episode_over:
        episode_over, reward = dm.next_turn(
            record_training_data=False,
            record_training_data_for_user=False,
        )
    hist = dm.state_tracker.dialog_history_dictionaries()
    return {
        "goal": goal,
        "turns": copy.deepcopy(hist),
        "outcome": {
            "reward": float(reward),
            "turn_count": int(dm.state_tracker.turn_count),
            "success": bool(reward > 0),
        },
        "seed": rng_seed,
    }


def main():
    ckpt_path = pick_best_checkpoint()
    print("loading", ckpt_path)
    dm, agent = build_dialog_manager(max_turn=30)
    agent.load(ckpt_path)

    # try many seeds, keep all successes
    successes = []
    n_seeds = 200
    seeds = list(range(1, n_seeds + 1))
    for s in seeds:
        try:
            r = rollout(dm, agent, s)
        except Exception as e:
            print("seed %d errored: %s" % (s, e))
            continue
        if r["outcome"]["success"]:
            successes.append(r)
    print("successful seeds: %d / %d" % (len(successes), n_seeds))

    if not successes:
        print("no success found — saving the shortest dialog instead")
        # fallback: rollout once with seed 1 and take it
        fallback = rollout(dm, agent, 1)
        best = fallback
    else:
        # Prefer dialogs that exercise the policy (request + inform variety),
        # not the trivial 2-turn ones, and within those pick the shortest.
        rich = [r for r in successes if r["outcome"]["turn_count"] >= 6]
        pool = rich if rich else successes
        pool.sort(key=lambda r: (r["outcome"]["turn_count"], -r["outcome"]["reward"]))
        best = pool[0]

    nlg_model = agent.nlg_model
    text = format_dialog_en(best["goal"], best["turns"], best["outcome"], nlg_model)
    header = (
        "Best demonstration dialog — DDQ agent (CamRest676)\n"
        "Checkpoint: %s\nSeed: %d\n%s\n"
        % (os.path.basename(ckpt_path), best["seed"], "=" * 60)
    )
    full = header + "\n" + text + "\n"

    txt_path = os.path.join(OUT, "best_dialog.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    print("saved", txt_path)

    md_path = os.path.join(OUT, "best_dialog.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Best demonstration dialog (DDQ on CamRest676)\n\n")
        f.write("- Checkpoint: `%s`\n" % os.path.basename(ckpt_path))
        f.write("- Seed: `%d`\n" % best["seed"])
        f.write("- Result: **%s** (reward=%s, turns=%s)\n\n"
                % ("SUCCESS" if best["outcome"]["success"] else "FAIL",
                   best["outcome"]["reward"], best["outcome"]["turn_count"]))
        f.write("```\n" + text + "\n```\n")
    print("saved", md_path)

    # render as a slide-friendly figure
    nlines = full.count("\n") + 1
    fig_h = max(5.0, min(12.0, 0.22 * nlines))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")
    ax.set_title("Best demonstration dialog — DDQ (CamRest676)",
                 fontsize=12, pad=10)
    ax.text(0.02, 0.98, full, transform=ax.transAxes,
            fontsize=9, family="monospace", va="top", ha="left")
    fig.tight_layout()
    png_path = os.path.join(OUT, "best_dialog.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", png_path)


if __name__ == "__main__":
    main()
