# Deep Dyna-Q for CamRest676

A task-oriented dialog system that learns a restaurant-recommendation policy with **Deep Dyna-Q (DDQ)**: a Deep Q-Network agent trained jointly against a rule-based user simulator and a learned world model. Ported from the original movie-ticketing reference implementation to the **CamRest676** Cambridge-restaurant domain.

## What this project does

The agent's job is to help a user find a restaurant by:

1. Eliciting the user's constraints (area, food, pricerange),
2. Looking up matches in a 110-restaurant knowledge base,
3. Answering the user's information requests (name / phone / address / postcode),
4. Closing the dialog when the user is satisfied.

A dialog is **successful** only when the agent answers every slot the user asked for, reports a matching restaurant, and the matched restaurant's constraint values agree with the user's goal. Failures are penalised; per-turn cost is `−1`; success rewards `+2·max_turn`.

## Architecture

```
                          ┌─────────────────┐
                          │  User Simulator │ (rule-based or learned)
                          └────────┬────────┘
                                   │ user_act
                                   ▼
  ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │   Knowledge Base │◀────│   State Tracker  │────▶│   Agent (DQN)   │
  │  (110 restaurants)│     │  (slot fill +    │     │  + world model  │
  └──────────────────┘     │   KB lookup)     │     │  + replay pool  │
                          └────────▲─────────┘     └────────┬────────┘
                                   │ sys_act               │
                                   └───────────────────────┘
```

Components (`deep_dialog/`):

| Module          | Role                                                            |
|-----------------|-----------------------------------------------------------------|
| `agents/`       | `AgentDQN` (learned) + rule baselines (`RequestBasicsAgent`, …) |
| `usersims/`     | `RuleSimulator` (handcrafted) and `ModelBasedSimulator` (world model) |
| `dialog_system/`| `DialogManager`, `StateTracker`, `KBHelper`                     |
| `qlearning/`    | DQN network + target network                                    |
| `nlg/`, `nlu/`  | Template-only by default; LSTM hooks available but unused       |
| `data/camrest676/` | KB, user goals, slot/act vocabularies derived from CamRest676 |

The action space is the cross product of dialog acts × slots:

- `inform(slot=PLACEHOLDER)` for each of `{name, phone, address, postcode, area, food, pricerange, taskcomplete}`,
- `request(slot=UNK)` for each of `{area, food, pricerange}`,
- and the bare acts `confirm_question`, `confirm_answer`, `thanks`, `deny`.

The state vector concatenates one-hot encodings of the last user act, agent act, current slot fills, turn count, and KB match counts (dimensions computed in `agent_dqn.py:prepare_state_representation`).

## Installation

Python 3.10+ with PyTorch.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy matplotlib
```

(Tested on Python 3.13 with PyTorch 2.9, but any recent combination works — there's nothing version-specific.)

## Quick start: rule-baseline smoke test

```powershell
python run_camrest.py --agt 5 --episodes 50 --max_turn 30
```

`--agt 5` is `RequestBasicsAgent`, a handcrafted rule policy. Expected output: ≈50–60 % success after the warm-start fix (see below). This confirms the dialog manager, KB lookup, user simulator, and template NLG are all wired up correctly.

Other rule agents available via `--agt`:

| `--agt` | Agent | Behavior |
|---|---|---|
| 1 | InformAgent | Iterates through slots informing placeholders |
| 2 | RequestAllAgent | Asks for every requestable slot, then `thanks` |
| 3 | RandomAgent | Uniform-random over feasible actions |
| 4 | EchoAgent | Informs whatever the user just requested |
| 5 | RequestBasicsAgent | Ask constraints → answer requests → inform(taskcomplete) → thanks |
| 9 | AgentDQN | Learned DDQ policy |

## Training the DDQ agent

```powershell
python run_camrest.py `
    --agt 9 `
    --episodes 200 `
    --warm_start 1 --warm_start_epochs 100 `
    --simulation_epoch_size 50 `
    --planning_steps 4 `
    --max_turn 30
```

Key knobs:

| Flag | Meaning | Default |
|---|---|---|
| `--episodes` | RL episodes after warm-start | 50 |
| `--warm_start_epochs` | Episodes the rule policy fills the replay pool | 100 |
| `--simulation_epoch_size` | Evaluation rollouts per training episode | 50 |
| `--planning_steps` | Dyna-Q planning rollouts via the world model | 4 |
| `--train_world_model` | 1 to update the world model alongside the agent | 1 |
| `--batch_size` | Replay sample size | 16 |
| `--gamma` | Discount factor | 0.9 |
| `--epsilon` | ε-greedy exploration | 0.0 (raise to ≈0.1 for more exploration) |
| `--max_turn` | Max turns before forced failure | 20 |

Checkpoints land in `deep_dialog/checkpoints/agt_9_<best_epoch>_<cur_epoch>_<rate>.pkl` every 10 episodes. `agt_9_performance_records.json` logs success-rate / avg-turns / avg-reward per evaluation point.

## Plotting the learning curve

```powershell
python draw_learning_curve.py --cmd 0 --result_file deep_dialog/checkpoints/agt_9_performance_records.json
```

`--cmd 1` prints the raw `episode<TAB>success<TAB>turns<TAB>reward` table instead.

## The warm-start bug (and the fix)

When this was first ported from the movie domain to CamRest676, the DDQ agent never learned: success rate stayed at 0 across 200 RL episodes.

**Root cause.** In CamRest, every user goal asks the agent to inform back at least one of `phone / address / postcode`. The original warm-start `rule_policy` (in `agent_dqn.py`) only:

1. Requested the three informable slots,
2. Issued `inform(taskcomplete)`,
3. Said `thanks`.

It never **answered the user's `request_slots`**, so the user simulator's success check in `usersim_rule.py:response_thanks` always failed (`rest_slots` was non-empty at the end of every dialog). The replay pool ended up 100 % negative reward, leaving the DQN with no signal to climb.

**Fix.** Insert an "answer pending user requests" phase into the rule policy, driven off `state['current_slots']['request_slots']` (which the state tracker prunes as the agent informs each slot):

```
phase 0: request area, food, pricerange   (user fills constraints)
phase 1: inform each pending user request (phone / address / postcode …)
phase 2: inform(taskcomplete)             (triggers constraint check)
phase 3: thanks                           (user evaluates → success)
```

Changes are localised to two files: `deep_dialog/agents/agent_dqn.py` (DQN warm-start) and `deep_dialog/agents/agent_baselines.py` (matching fix for `RequestBasicsAgent` so the rule baseline reflects the same logic).

### Results (200 episodes, warm-start 100, planning 4)

| Metric | Before fix | After fix |
|---|---|---|
| Warm-start success rate | 0 % | **66 %** |
| Eval peak success | 0 % | **70 %** (ep 150) |
| Rule baseline (`--agt 5`) | 0 / 50 | **29 / 50 (58 %)** |
| Avg reward (final 20 evals) | −30 | −14.5 |

The DQN policy itself still oscillates (final-20 mean 26 %, peak 70 %) — converging it cleanly is a hyperparameter exercise (longer training, slower target-net updates, best-checkpoint reloading), not a setup problem.

## Project layout

```
.
├── run_camrest.py                 # main entry point
├── draw_learning_curve.py         # plot success-rate / reward curves
├── deep_dialog/
│   ├── dialog_config.py           # domain slots, rewards, feasible actions
│   ├── agents/
│   │   ├── agent_dqn.py           # AgentDQN (DDQ policy)
│   │   └── agent_baselines.py     # rule-based agents
│   ├── usersims/
│   │   ├── usersim_rule.py        # handcrafted user
│   │   └── usersim_model.py       # world model (also a learned user)
│   ├── dialog_system/
│   │   ├── dialog_manager.py
│   │   ├── state_tracker.py
│   │   └── kb_helper.py           # KB lookup + slot fill
│   ├── qlearning/                 # DQN network
│   ├── nlg/, nlu/                 # template NLG + no-op NLU
│   └── data/camrest676/
│       ├── build_camrest_data.py  # regenerate pickles from raw CamRest676
│       ├── CamRest676.json, CamRestDB.json, CamRestOTGY.json
│       └── camrest_kb.p, camrest_user_goals.p, …
```

## Regenerating the dataset

The pickled KB and user goals are already in `deep_dialog/data/camrest676/`. To rebuild them from the raw CamRest676 release (also in that directory):

```powershell
cd deep_dialog\data\camrest676
python build_camrest_data.py
```

## Citation

Dataset — CamRest676 (Wen et al., Cambridge Dialogue Systems Group, 2016):

```bibtex
@article{wenN2N16,
  author  = {Wen, Tsung-Hsien and Vandyke, David and Mrk{\v{s}}i\'c, Nikola and
             Ga{\v{s}}i\'c, Milica and Rojas-Barahona, Lina M. and Su, Pei-Hao and
             Ultes, Stefan and Young, Steve},
  title   = {A Network-based End-to-End Trainable Task-oriented Dialogue System},
  journal = {arXiv:1604.04562},
  year    = {2016}
}
```

Algorithm — Deep Dyna-Q (Peng et al., 2018):

```bibtex
@inproceedings{peng2018deep,
  author    = {Peng, Baolin and Li, Xiujun and Gao, Jianfeng and Liu, Jingjing and
               Wong, Kam-Fai and Su, Shang-Yu},
  title     = {Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning},
  booktitle = {ACL},
  year      = {2018}
}
```

## License

Code is provided as-is for research and educational use. The CamRest676 dataset is © Cambridge Dialogue Systems Group, 2016, and is included under its original terms (see `deep_dialog/data/camrest676/README.json`).
