import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from gridworld import GridWorldEnv
from dp import DPAgent

def run_policy_iteration(env, theta=1e-6):
    agent = DPAgent(env)
    n_iter = agent.policy_iteration(theta=theta)
    print(f"[Policy Iteration] converge in {n_iter} loops (theta={theta})")
    return agent


def run_value_iteration(env, theta=1e-6):
    agent = DPAgent(env)
    n_iter = agent.value_iteration(theta=theta)
    print(f"[Value Iteration] converge in {n_iter} loops (theta={theta})")
    return agent

def value_dict_to_grid(V, env):
    grid = np.full((env.width, env.height), np.nan)
    for (x, y), v in V.items():
        grid[x, y] = v
    return grid


def plot_value_on_ax(ax, V, env, title):
    grid = value_dict_to_grid(V, env)
    masked = np.ma.masked_invalid(grid)  

    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")      

    im = ax.imshow(masked, origin="lower", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for x in range(env.width):
        for y in range(env.height):
            if not np.isnan(grid[x, y]):
                ax.text(y, x, f"{grid[x, y]:.2f}", ha="center", va="center",
                         color="white", fontsize=8)

    ax.set_title(title)
    ax.set_xticks(range(env.height))
    ax.set_yticks(range(env.width))


def plot_policy_on_ax(ax, policy_dict, env, title):
    ax.set_xlim(-0.5, env.height - 0.5)
    ax.set_ylim(-0.5, env.width - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.height))
    ax.set_yticks(range(env.width))
    ax.grid(True, linewidth=0.5, color="lightgray")

    for (x, y) in env.walls:
        ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="dimgray"))
    for (x, y) in env.lakes:
        ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="lightblue"))
    ex, ey = env.exit
    ax.add_patch(plt.Rectangle((ey - 0.5, ex - 0.5), 1, 1, color="lightgreen"))

    for (x, y), a_id in policy_dict.items():
        dx, dy = env.action[a_id]
        ax.arrow(y, x, dy * 0.3, dx * 0.3,
                  head_width=0.15, head_length=0.15,
                  fc="black", ec="black", length_includes_head=True)

    ax.set_title(title)


def main():
    env = GridWorldEnv()

    pi_agent = run_policy_iteration(env)
    vi_agent = run_value_iteration(env)

    pi_policy_dict = pi_agent.extract_policy()
    vi_policy_dict = vi_agent.extract_policy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    plot_value_on_ax(axes[0, 0], pi_agent.V, env, "V*  (Policy Iteration)")
    plot_policy_on_ax(axes[0, 1], pi_policy_dict, env, "pi*  (Policy Iteration)")

    plot_value_on_ax(axes[1, 0], vi_agent.V, env, "V*  (Value Iteration)")
    plot_policy_on_ax(axes[1, 1], vi_policy_dict, env, "pi*  (Value Iteration)")

    plt.tight_layout()
    plt.savefig("dp_results.png", dpi=150)
    plt.show()

    pi_policy_dict = pi_agent.extract_policy()
    vi_policy_dict = vi_agent.extract_policy()

if __name__ == "__main__":
    main()