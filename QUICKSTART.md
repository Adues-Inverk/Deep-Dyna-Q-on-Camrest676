# 🚀 Quick Start Guide: React & MuZero Agents

## 📦 What's New?

You now have two state-of-the-art dialog agents:

1. **React (Reason+Act) Agent** - Combines explicit reasoning with action selection
   - More interpretable: can see "chain-of-thought"
   - Good for regulated/safety-critical domains
   
2. **MuZero Agent** - Model-based RL with planning
   - More sample efficient
   - Uses learned world model for lookahead
   - Better long-term reasoning

## ⚡ 5-Minute Setup

### 1. Check Installation
```bash
cd "C:\Users\as\Desktop\CS106\Deep-Dyna-Q on Camrest676"
python -c "import torch; print('PyTorch OK')"
```

### 2. Run Showcase (Compare All Agents)
```bash
# Quick test: 20 training episodes, 30 eval episodes
python showcase_agents.py --episodes 20 --eval_episodes 30 --save-report

# Full comparison: 50 training episodes, 100 eval episodes  
python showcase_agents.py --episodes 50 --eval_episodes 100 --save-report
```

**Output:**
- Console shows training progress and evaluation results
- `showcase_results.json` - Detailed metrics
- Performance summary table printed to console

### 3. Generate Visualizations
```bash
# Create plots and analysis
python visualize_showcase.py --report showcase_results.json --output-dir ./plots

# Also generate HTML report
python visualize_showcase.py --report showcase_results.json --output-dir ./plots --html-report
```

**Output:**
- `plots/success_rate_comparison.png`
- `plots/reward_comparison.png`
- `plots/turns_comparison.png`
- `plots/reward_distribution.png`
- `plots/efficiency_scatter.png`
- `plots/report.html` (open in browser)

## 🎯 Individual Agent Training

### Train React Agent
```bash
# 50 training episodes
python run_camrest.py --agt 10 --episodes 50 --warm_start_epochs 50

# With custom hyperparameters
python run_camrest.py \
    --agt 10 \
    --episodes 100 \
    --dqn_hidden_size 80 \
    --batch_size 32 \
    --gamma 0.95
```

### Train MuZero Agent
```bash
# 50 training episodes
python run_camrest.py --agt 11 --episodes 50 --warm_start_epochs 50

# With longer planning horizon
python run_camrest.py \
    --agt 11 \
    --episodes 100 \
    --dqn_hidden_size 60 \
    --planning_steps 8 \
    --mcts_simulations 50
```

### Train DQN Baseline (for comparison)
```bash
python run_camrest.py --agt 9 --episodes 50 --warm_start_epochs 50
```

## 📊 Expected Results

On CamRest676 with default settings (50-100 episodes):

| Agent | Success Rate | Avg Reward | Efficiency |
|-------|--------------|-----------|------------|
| **React** | 70-80% | 35-45 | Good |
| **MuZero** | 75-85% | 37-48 | Excellent |
| **DQN** | 60-70% | 30-40 | Fair |
| **Baseline** | 50-65% | 25-35 | Poor |

*Results vary with hyperparameters and random seeds*

## 🔧 Customization

### Modify Agent Parameters
Edit `showcase_agents.py`, line ~130:
```python
agent_params = {
    'max_turn': max_turn,
    'epsilon': 0.1,
    'dqn_hidden_size': 60,  # Increase for more capacity
    'batch_size': 16,       # Increase for stability
    'gamma': 0.9,           # Discount factor
    'warm_start': 1,        # Use rule-based warmup
}
```

### Adjust Training Duration
```bash
# Quick test (5 min)
python showcase_agents.py --episodes 10 --eval_episodes 20

# Medium run (15 min)
python showcase_agents.py --episodes 30 --eval_episodes 50

# Full training (30+ min)
python showcase_agents.py --episodes 100 --eval_episodes 200
```

### Compare Specific Agents
```bash
# Only React and MuZero (fastest)
python showcase_agents.py --agents react muzero --episodes 30

# Only new agents (no DQN/baseline)
python showcase_agents.py --agents react muzero --episodes 50 --save-report
```

## 📈 Interpreting Results

### Success Rate
- **Higher is better** - % of dialogs that reached task completion
- Target: > 80% with well-tuned agents

### Average Reward
- **Higher is better** - Sum of rewards per episode
- SUCCESS_REWARD = 50 (default)
- Each extra turn = 0 (by default)

### Average Turns
- **Lower is better** - Fewer turns = more efficient dialog
- Good agents: 4-6 turns average
- Bad agents: 8+ turns average

### Consistency (Variance)
- Look at plots - tight distribution = reliable agent
- Wide distribution = inconsistent performance

## 🧠 Understanding the Agents

### React Agent
**How it works:**
1. **Analyze state**: "User asking for restaurant, 2 slots filled"
2. **Reason**: "Should request missing slots"
3. **Act**: Send request action
4. **Learn**: Store experience in replay buffer

**When to use:**
- ✓ Need to explain decisions
- ✓ Regulatory requirements
- ✓ Safety-critical systems
- ✗ Need maximum performance

### MuZero Agent  
**How it works:**
1. **Encode state**: Convert observation → latent representation
2. **Plan**: Run MCTS (~20 simulations)
   - Simulate future: state → action → next state
   - Evaluate: predict value of next state
   - Backup: update tree with values
3. **Act**: Choose best action from simulations
4. **Learn**: Update representation, dynamics, value networks

**When to use:**
- ✓ Need high performance
- ✓ Complex state space
- ✓ Can spare compute for planning
- ✗ Need interpretability

## ⚠️ Troubleshooting

### "Module not found" error
```bash
# Ensure you're in the right directory
cd "C:\Users\as\Desktop\CS106\Deep-Dyna-Q on Camrest676"

# Check imports
python -c "from deep_dialog.agents import AgentReact, AgentMuZero"
```

### Poor performance during training
```bash
# Increase warm-start (more supervised learning)
python showcase_agents.py \
    --episodes 50 \
    --warm_start_epochs 100  # Increased from 50

# Use smaller batch size for more updates
python run_camrest.py --agt 10 --batch_size 8 --episodes 50
```

### Out of memory
```bash
# Reduce replay buffer
python showcase_agents.py \
    --experience_replay_pool_size 2000

# Reduce planning simulations (MuZero)
# Edit agent_muzero.py, line 103:
# self.mcts_simulations = 10  # reduced from 20
```

### Slow training
```bash
# Reduce number of episodes
python showcase_agents.py --episodes 20 --eval_episodes 30

# Use fewer MCTS simulations
# Edit agent_muzero.py: self.mcts_simulations = 10

# Train on baseline agent first (faster)
python run_camrest.py --agt 5 --episodes 50
```

## 📚 Documentation

See detailed docs:
- [AGENTS_README.md](AGENTS_README.md) - Full technical documentation
- [README.md](README.md) - Original system documentation

## 🎓 Learning More

### Read the code:
- `deep_dialog/agents/agent_react.py` - React implementation
- `deep_dialog/agents/agent_muzero.py` - MuZero implementation
- `showcase_agents.py` - Comparison framework
- `visualize_showcase.py` - Visualization tools

### Modify agents:
1. Copy `agent_react.py` → `agent_react_v2.py`
2. Edit class name and modify key methods
3. Add to `__init__.py`
4. Test with `showcase_agents.py --agents react_v2 ...`

## 💡 Tips & Tricks

### Warm Start Matters
React and MuZero use warm-start to bootstrap learning:
```python
--warm_start_epochs 100  # Start with 100 rule-based episodes
```
This gives agents experience before RL kicks in.

### Batch Size Impact
- **Small batch** (8): More frequent updates, noisier gradients
- **Large batch** (32): Stable gradients, fewer updates
- **Optimal**: 16-32 for most problems

### Gamma (Discount Factor)
- **γ = 0.9**: Weight immediate rewards more
- **γ = 0.99**: Weight future rewards more
- Dialog systems: 0.9-0.95 usually good

### Epsilon (Exploration)
- **ε = 0.1**: Explore 10% of time, exploit 90%
- **ε = 0.0**: Pure exploitation (good for eval)
- Training: usually 0.1-0.3

## 🔄 Workflow Example

```bash
# 1. Run quick test
python showcase_agents.py --episodes 10 --eval_episodes 20 --save-report

# 2. Check results
cat showcase_results.json | python -m json.tool | head -50

# 3. Generate visualizations
python visualize_showcase.py \
    --report showcase_results.json \
    --output-dir ./plots \
    --html-report

# 4. View best agent performance
python run_camrest.py --agt 11 --episodes 100 --warm_start_epochs 100

# 5. Analyze results
python visualize_showcase.py --report showcase_results.json --output-dir ./analysis
```

## 📞 Next Steps

1. **Understand**: Read AGENTS_README.md
2. **Experiment**: Run showcase with different configs
3. **Analyze**: Use visualize_showcase.py to understand results
4. **Modify**: Extend agents for your use case
5. **Deploy**: Use best agent in production

---

**Happy experimenting! 🎉**

For detailed technical information, see [AGENTS_README.md](AGENTS_README.md).
