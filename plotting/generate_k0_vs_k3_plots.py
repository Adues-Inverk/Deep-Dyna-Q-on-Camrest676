"""Generate side-by-side comparison plots for k=0 (baseline) vs k=3 (DDQ)"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

FIGURES_DIR = 'slides_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_performance_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load data for both k values
k0_data = load_performance_data('deep_dialog/checkpoints/sweep_k/k_0/agt_9_performance_records.json')
k3_data = load_performance_data('deep_dialog/checkpoints/sweep_k/k_3/agt_9_performance_records.json')

# Extract k=0 data
keylist_k0 = sorted(int(k) for k in k0_data['success_rate'].keys())
x_k0 = np.array(keylist_k0)
sr_k0 = np.array([k0_data['success_rate'][str(k)] for k in keylist_k0])
turns_k0 = np.array([k0_data['ave_turns'][str(k)] for k in keylist_k0])
reward_k0 = np.array([k0_data['ave_reward'][str(k)] for k in keylist_k0])

# Extract k=3 data
keylist_k3 = sorted(int(k) for k in k3_data['success_rate'].keys())
x_k3 = np.array(keylist_k3)
sr_k3 = np.array([k3_data['success_rate'][str(k)] for k in keylist_k3])
turns_k3 = np.array([k3_data['ave_turns'][str(k)] for k in keylist_k3])
reward_k3 = np.array([k3_data['ave_reward'][str(k)] for k in keylist_k3])

# ==================== Figure 1: Side-by-side Learning Curves ====================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Deep Dyna-Q vs Baseline: Learning Dynamics Comparison (Agent 9)', fontsize=16, fontweight='bold')

# --- Success Rate ---
# K=0 (Baseline)
ax = axes[0, 0]
ax.plot(x_k0, sr_k0, 'o-', color='#2E86AB', linewidth=2.5, markersize=5, label='K=0 (Baseline)')
ax.fill_between(x_k0, sr_k0 * 0.95, sr_k0 * 1.05, alpha=0.2, color='#2E86AB')
ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_title('K=0: Baseline (No Planning)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])
ax.legend(loc='lower right')

# K=3 (DDQ)
ax = axes[0, 1]
ax.plot(x_k3, sr_k3, 'o-', color='#06A77D', linewidth=2.5, markersize=5, label='K=3 (DDQ)')
ax.fill_between(x_k3, sr_k3 * 0.95, sr_k3 * 1.05, alpha=0.2, color='#06A77D')
ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_title('K=3: Deep Dyna-Q (3-Step Planning)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])
ax.legend(loc='lower right')

# Comparison overlay
ax = axes[0, 2]
ax.plot(x_k0, sr_k0, 'o-', color='#2E86AB', linewidth=2.5, markersize=5, alpha=0.7, label='K=0 (Baseline)')
ax.plot(x_k3, sr_k3, 's-', color='#06A77D', linewidth=2.5, markersize=5, alpha=0.7, label='K=3 (DDQ)')
ax.fill_between(x_k0, sr_k0 * 0.95, sr_k0 * 1.05, alpha=0.15, color='#2E86AB')
ax.fill_between(x_k3, sr_k3 * 0.95, sr_k3 * 1.05, alpha=0.15, color='#06A77D')
ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_title('Success Rate: Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])
ax.legend(loc='lower right')

# --- Average Turns ---
# K=0 (Baseline)
ax = axes[1, 0]
ax.plot(x_k0, turns_k0, 's-', color='#A23B72', linewidth=2.5, markersize=5, label='K=0 (Baseline)')
ax.fill_between(x_k0, turns_k0 * 0.95, turns_k0 * 1.05, alpha=0.2, color='#A23B72')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Turns', fontsize=11, fontweight='bold')
ax.set_title('K=0: Dialogue Length', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# K=3 (DDQ)
ax = axes[1, 1]
ax.plot(x_k3, turns_k3, 's-', color='#F18F01', linewidth=2.5, markersize=5, label='K=3 (DDQ)')
ax.fill_between(x_k3, turns_k3 * 0.95, turns_k3 * 1.05, alpha=0.2, color='#F18F01')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Turns', fontsize=11, fontweight='bold')
ax.set_title('K=3: Dialogue Length', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Comparison overlay
ax = axes[1, 2]
ax.plot(x_k0, turns_k0, 's-', color='#A23B72', linewidth=2.5, markersize=5, alpha=0.7, label='K=0 (Baseline)')
ax.plot(x_k3, turns_k3, '^-', color='#F18F01', linewidth=2.5, markersize=5, alpha=0.7, label='K=3 (DDQ)')
ax.fill_between(x_k0, turns_k0 * 0.95, turns_k0 * 1.05, alpha=0.15, color='#A23B72')
ax.fill_between(x_k3, turns_k3 * 0.95, turns_k3 * 1.05, alpha=0.15, color='#F18F01')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Turns', fontsize=11, fontweight='bold')
ax.set_title('Dialogue Length: Comparison (Lower=Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'k0_vs_k3_learning_curves.png'), dpi=300, bbox_inches='tight')
print("[SAVED] k0_vs_k3_learning_curves.png")
plt.close()

# ==================== Figure 2: Reward Comparison ====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Reward Dynamics: K=0 (Baseline) vs K=3 (Deep Dyna-Q)', fontsize=14, fontweight='bold')

# K=0 (Baseline)
ax = axes[0]
ax.plot(x_k0, reward_k0, '^-', color='#C1121F', linewidth=2.5, markersize=5, label='K=0 (Baseline)')
ax.fill_between(x_k0, reward_k0 * 0.95, reward_k0 * 1.05, alpha=0.2, color='#C1121F')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
ax.set_title('K=0: Reward Over Training', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

# K=3 (DDQ)
ax = axes[1]
ax.plot(x_k3, reward_k3, '^-', color='#06A77D', linewidth=2.5, markersize=5, label='K=3 (DDQ)')
ax.fill_between(x_k3, reward_k3 * 0.95, reward_k3 * 1.05, alpha=0.2, color='#06A77D')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
ax.set_title('K=3: Reward Over Training', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

# Comparison overlay
ax = axes[2]
ax.plot(x_k0, reward_k0, '^-', color='#C1121F', linewidth=2.5, markersize=5, alpha=0.7, label='K=0 (Baseline)')
ax.plot(x_k3, reward_k3, '^-', color='#06A77D', linewidth=2.5, markersize=5, alpha=0.7, label='K=3 (DDQ)')
ax.fill_between(x_k0, reward_k0 * 0.95, reward_k0 * 1.05, alpha=0.15, color='#C1121F')
ax.fill_between(x_k3, reward_k3 * 0.95, reward_k3 * 1.05, alpha=0.15, color='#06A77D')
ax.set_xlabel('Training Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
ax.set_title('Reward: Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'k0_vs_k3_reward.png'), dpi=300, bbox_inches='tight')
print("[SAVED] k0_vs_k3_reward.png")
plt.close()

# ==================== Figure 3: Performance Summary Bar Charts ====================
# Compute summary metrics
peak_sr_k0 = np.max(sr_k0)
final_sr_k0 = np.mean(sr_k0[-20:]) if len(sr_k0) >= 20 else np.mean(sr_k0)
stability_k0 = final_sr_k0 / peak_sr_k0 if peak_sr_k0 > 0 else 0

peak_sr_k3 = np.max(sr_k3)
final_sr_k3 = np.mean(sr_k3[-20:]) if len(sr_k3) >= 20 else np.mean(sr_k3)
stability_k3 = final_sr_k3 / peak_sr_k3 if peak_sr_k3 > 0 else 0

peak_reward_k0 = np.max(reward_k0)
final_reward_k0 = np.mean(reward_k0[-20:]) if len(reward_k0) >= 20 else np.mean(reward_k0)

peak_reward_k3 = np.max(reward_k3)
final_reward_k3 = np.mean(reward_k3[-20:]) if len(reward_k3) >= 20 else np.mean(reward_k3)

peak_turns_k0 = np.min(turns_k0)  # Lower is better
final_turns_k0 = np.mean(turns_k0[-20:])

peak_turns_k3 = np.min(turns_k3)  # Lower is better
final_turns_k3 = np.mean(turns_k3[-20:])

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Performance Summary: K=0 (Baseline) vs K=3 (Deep Dyna-Q)', fontsize=14, fontweight='bold')

methods = ['K=0\n(Baseline)', 'K=3\n(DDQ)']
colors_baseline = '#2E86AB'
colors_ddq = '#06A77D'
colors = [colors_baseline, colors_ddq]

width = 0.35
x_pos = np.arange(len(methods))

# --- Success Rate ---
ax = axes[0]
peak_vals = [peak_sr_k0, peak_sr_k3]
final_vals = [final_sr_k0, final_sr_k3]

bars1 = ax.bar(x_pos - width/2, peak_vals, width, label='Peak', color=colors, alpha=0.8)
bars2 = ax.bar(x_pos + width/2, final_vals, width, label='Final-20 Avg', color=colors, alpha=0.4)

ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# Add value labels
for i, (peak, final) in enumerate(zip(peak_vals, final_vals)):
    ax.text(i - width/2, peak + 0.03, f'{peak:.1%}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, final + 0.03, f'{final:.1%}', ha='center', fontsize=10, fontweight='bold')

# --- Average Turns (Lower is Better) ---
ax = axes[1]
peak_turns_vals = [peak_turns_k0, peak_turns_k3]
final_turns_vals = [final_turns_k0, final_turns_k3]

bars1 = ax.bar(x_pos - width/2, peak_turns_vals, width, label='Best', color=colors, alpha=0.8)
bars2 = ax.bar(x_pos + width/2, final_turns_vals, width, label='Final-20 Avg', color=colors, alpha=0.4)

ax.set_ylabel('Average Turns', fontsize=12, fontweight='bold')
ax.set_title('Dialogue Length (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (peak, final) in enumerate(zip(peak_turns_vals, final_turns_vals)):
    ax.text(i - width/2, peak + 0.5, f'{peak:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, final + 0.5, f'{final:.1f}', ha='center', fontsize=10, fontweight='bold')

# --- Average Reward ---
ax = axes[2]
peak_reward_vals = [peak_reward_k0, peak_reward_k3]
final_reward_vals = [final_reward_k0, final_reward_k3]

bars1 = ax.bar(x_pos - width/2, peak_reward_vals, width, label='Peak', color=colors, alpha=0.8)
bars2 = ax.bar(x_pos + width/2, final_reward_vals, width, label='Final-20 Avg', color=colors, alpha=0.4)

ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
ax.set_title('Reward Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Add value labels
for i, (peak, final) in enumerate(zip(peak_reward_vals, final_reward_vals)):
    ax.text(i - width/2, peak + 0.5, f'{peak:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, final + 0.5, f'{final:.2f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'k0_vs_k3_summary.png'), dpi=300, bbox_inches='tight')
print("[SAVED] k0_vs_k3_summary.png")
plt.close()

# ==================== Print Summary Statistics ====================
print("\n" + "="*70)
print("PERFORMANCE COMPARISON: K=0 (BASELINE) vs K=3 (DEEP DYNA-Q)")
print("="*70)

print("\n--- SUCCESS RATE ---")
print(f"K=0 Baseline: Peak={peak_sr_k0:.1%}, Final-20={final_sr_k0:.1%}, Stability={stability_k0:.1%}")
print(f"K=3 DDQ:      Peak={peak_sr_k3:.1%}, Final-20={final_sr_k3:.1%}, Stability={stability_k3:.1%}")
improvement_sr = ((final_sr_k3 - final_sr_k0) / final_sr_k0 * 100) if final_sr_k0 > 0 else 0
print(f"Improvement:  {improvement_sr:+.1f}%")

print("\n--- AVERAGE TURNS ---")
print(f"K=0 Baseline: Best={peak_turns_k0:.1f}, Final-20={final_turns_k0:.1f}")
print(f"K=3 DDQ:      Best={peak_turns_k3:.1f}, Final-20={final_turns_k3:.1f}")
improvement_turns = ((final_turns_k0 - final_turns_k3) / final_turns_k0 * 100) if final_turns_k0 > 0 else 0
print(f"Improvement:  {improvement_turns:+.1f}% (positive = shorter dialogues)")

print("\n--- AVERAGE REWARD ---")
print(f"K=0 Baseline: Peak={peak_reward_k0:.2f}, Final-20={final_reward_k0:.2f}")
print(f"K=3 DDQ:      Peak={peak_reward_k3:.2f}, Final-20={final_reward_k3:.2f}")
improvement_reward = ((final_reward_k3 - final_reward_k0) / abs(final_reward_k0) * 100) if final_reward_k0 != 0 else 0
print(f"Improvement:  {improvement_reward:+.1f}%")

print("\n" + "="*70)
print(f"[DONE] All comparison plots generated successfully!")
print(f"  Location: {os.path.abspath(FIGURES_DIR)}")
print("  Files:")
print("    - k0_vs_k3_learning_curves.png")
print("    - k0_vs_k3_reward.png")
print("    - k0_vs_k3_summary.png")
print("="*70)
