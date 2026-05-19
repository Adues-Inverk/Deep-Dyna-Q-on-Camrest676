#!/usr/bin/env python3
"""
Visualization and Analysis Script for Agent Showcase

Generates detailed analysis, plots, and reports from showcase results.

Usage:
    python visualize_showcase.py --report showcase_results.json --output-dir ./plots
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_report(report_path):
    """Load showcase results report"""
    with open(report_path, 'r') as f:
        return json.load(f)


def print_detailed_analysis(report):
    """Print detailed analysis of results"""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    results = report['results']
    
    # Success rate analysis
    print("\n📊 SUCCESS RATE ANALYSIS")
    print("─" * 40)
    agents = sorted(results.keys(), key=lambda a: results[a]['success_rate'], reverse=True)
    for i, agent in enumerate(agents, 1):
        sr = results[agent]['success_rate']
        print(f"{i}. {agent:<20} {sr:>6.1%}")
    
    # Reward analysis
    print("\n💰 REWARD ANALYSIS")
    print("─" * 40)
    agents_by_reward = sorted(results.keys(), key=lambda a: results[a]['avg_reward'], reverse=True)
    for i, agent in enumerate(agents_by_reward, 1):
        rewards = results[agent]['avg_reward']
        print(f"{i}. {agent:<20} {rewards:>8.2f}")
    
    # Turn efficiency analysis
    print("\n⏱️  TURN EFFICIENCY ANALYSIS")
    print("─" * 40)
    agents_by_turns = sorted(results.keys(), key=lambda a: results[a]['avg_turns'])
    for i, agent in enumerate(agents_by_turns, 1):
        turns = results[agent]['avg_turns']
        print(f"{i}. {agent:<20} {turns:>6.2f} turns")
    
    # Consistency analysis (variance)
    print("\n📈 CONSISTENCY ANALYSIS (Reward Variance)")
    print("─" * 40)
    consistency = {}
    for agent, data in results.items():
        rewards = data['dialog_rewards']
        if len(rewards) > 1:
            variance = statistics.variance(rewards)
            stdev = statistics.stdev(rewards)
            consistency[agent] = (variance, stdev)
        else:
            consistency[agent] = (0, 0)
    
    agents_by_consistency = sorted(consistency.items(), key=lambda x: x[1][1])
    for i, (agent, (var, std)) in enumerate(agents_by_consistency, 1):
        print(f"{i}. {agent:<20} Std Dev: {std:>6.2f}")
    
    # Agent characteristics
    print("\n🎯 AGENT CHARACTERISTICS")
    print("─" * 40)
    for agent in sorted(results.keys()):
        print(f"\n{agent}:")
        data = results[agent]
        dialogs = data['dialog_rewards']
        print(f"  Total dialogs evaluated: {len(dialogs)}")
        print(f"  Reward range: [{min(dialogs):.1f}, {max(dialogs):.1f}]")
        print(f"  Reward mean: {statistics.mean(dialogs):.2f}")
        if len(dialogs) > 1:
            print(f"  Reward stdev: {statistics.stdev(dialogs):.2f}")
        
        lengths = data['dialog_lengths']
        print(f"  Turn range: [{min(lengths)}, {max(lengths)}]")
        print(f"  Turn mean: {statistics.mean(lengths):.2f}")


def create_comparison_plots(report, output_dir):
    """Create comparison plots"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots: matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    results = report['results']
    agents = sorted(results.keys())
    
    # Plot 1: Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    success_rates = [results[a]['success_rate'] * 100 for a in agents]
    bars = ax.bar(agents, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Agent Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=150)
    print(f"✓ Saved: success_rate_comparison.png")
    plt.close()
    
    # Plot 2: Average Reward Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_rewards = [results[a]['avg_reward'] for a in agents]
    bars = ax.bar(agents, avg_rewards, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Agent Average Reward Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=150)
    print(f"✓ Saved: reward_comparison.png")
    plt.close()
    
    # Plot 3: Average Turns Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_turns = [results[a]['avg_turns'] for a in agents]
    bars = ax.bar(agents, avg_turns, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('Average Turns', fontsize=12)
    ax.set_title('Agent Dialogue Length Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'turns_comparison.png'), dpi=150)
    print(f"✓ Saved: turns_comparison.png")
    plt.close()
    
    # Plot 4: Reward Distribution Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    reward_data = [results[a]['dialog_rewards'] for a in agents]
    bp = ax.boxplot(reward_data, labels=agents, patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Distribution Across Agents', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=150)
    print(f"✓ Saved: reward_distribution.png")
    plt.close()
    
    # Plot 5: Scatter plot: Success Rate vs Avg Turns
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, agent in enumerate(agents):
        sr = results[agent]['success_rate'] * 100
        turns = results[agent]['avg_turns']
        ax.scatter(turns, sr, s=300, alpha=0.6, label=agent)
        ax.annotate(agent, (turns, sr), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('Average Turns (Lower is Better)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Agent Efficiency: Success Rate vs Dialogue Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_scatter.png'), dpi=150)
    print(f"✓ Saved: efficiency_scatter.png")
    plt.close()


def generate_html_report(report, output_path):
    """Generate HTML report"""
    results = report['results']
    timestamp = report['timestamp']
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Showcase Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { font-size: 1.1em; opacity: 0.9; }
        .content { padding: 40px; }
        h2 {
            color: #333;
            margin-top: 30px;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #f9f9f9;
        }
        th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover { background: #f0f0f0; }
        .metric { margin: 15px 0; }
        .metric-label { font-weight: 600; color: #333; }
        .metric-value { font-size: 1.3em; color: #667eea; font-weight: bold; }
        .agent-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .agent-card h3 { color: #667eea; margin-bottom: 15px; }
        .badge {
            display: inline-block;
            padding: 5px 15px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-right: 10px;
        }
        .best { background: #4CAF50; }
        footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Agent Showcase Report</h1>
            <p class="subtitle">React (Reason+Act) vs MuZero on CamRest676</p>
        </header>
        <div class="content">
"""
    
    # Summary table
    html += """            <h2>📊 Performance Summary</h2>
            <table>
                <tr>
                    <th>Agent</th>
                    <th>Success Rate</th>
                    <th>Average Reward</th>
                    <th>Average Turns</th>
                </tr>
"""
    
    for agent in sorted(results.keys()):
        data = results[agent]
        html += f"""                <tr>
                    <td><strong>{agent}</strong></td>
                    <td>{data['success_rate']:.1%}</td>
                    <td>{data['avg_reward']:.2f}</td>
                    <td>{data['avg_turns']:.2f}</td>
                </tr>
"""
    
    html += """            </table>"""
    
    # Detailed results
    html += """            <h2>🔍 Detailed Agent Analysis</h2>
"""
    
    for agent in sorted(results.keys()):
        data = results[agent]
        html += f"""            <div class="agent-card">
                <h3>{agent}</h3>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{data['success_rate']:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Reward:</span>
                    <span class="metric-value">{data['avg_reward']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Turns:</span>
                    <span class="metric-value">{data['avg_turns']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Reward Range:</span>
                    <span class="metric-value">[{min(data['dialog_rewards']):.1f}, {max(data['dialog_rewards']):.1f}]</span>
                </div>
            </div>
"""
    
    # Footer
    html += f"""        </div>
        <footer>
            <p>Generated: {timestamp}</p>
            <p>CamRest676 Dialog System Benchmark</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize showcase results')
    parser.add_argument('--report', default='showcase_results.json',
                        help='Path to showcase results JSON file')
    parser.add_argument('--output-dir', default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--html-report', action='store_true',
                        help='Generate HTML report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.report):
        print(f"Error: Report file not found: {args.report}")
        return
    
    print("="*80)
    print("SHOWCASE RESULTS ANALYSIS")
    print("="*80)
    
    report = load_report(args.report)
    print(f"📋 Loaded report from: {args.report}")
    
    # Print analysis
    print_detailed_analysis(report)
    
    # Create plots
    print(f"\n📈 Generating comparison plots...")
    create_comparison_plots(report, args.output_dir)
    
    # Generate HTML report
    if args.html_report:
        print(f"\n📄 Generating HTML report...")
        html_path = os.path.join(args.output_dir, 'report.html')
        generate_html_report(report, html_path)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
