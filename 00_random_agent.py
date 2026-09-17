import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

env = gym.make("CartPole-v1")
returns = []  # returns

# 100 random episodes
for episode in tqdm(range(100), desc="Running Random Agent"):
    state, info = env.reset()
    episode_return = 0
    done = False
    
    while not done:
        # 0: left ; 1: right
        action = env.action_space.sample() 
        state, reward, terminated, truncated, info = env.step(action)
        
        episode_return += reward
        done = terminated or truncated
        
    returns.append(episode_return)

env.close()

mean_return = np.mean(returns)
std_return = np.std(returns)
print(f"\n[Baseline] Random Agent Performance (100 episodes): {mean_return:.2f} ± {std_return:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(returns, marker='o', linestyle='-', color='b', alpha=0.7)
plt.axhline(mean_return, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}')
plt.title("CartPole-v1: Random Agent Returns (100 Episodes)")
plt.xlabel("Episode")
plt.ylabel("Return (Total Reward)")
plt.legend()
plt.grid(True, alpha=0.3)

save_path = "figs/00_random_agent_baseline.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')