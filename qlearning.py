import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
EPISODES  = 5000

def q_learning(env, episodes=EPISODES, learningRate=0.8, discountRate=0.95, epsilon=0.1):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    all_states = []
    all_actions = []
    Q = np.zeros((state_size, action_size))
    rewards_per_episode = np.zeros(episodes)
    rng = np.random.default_rng(12314512561234)
    for episode in range(episodes):
        state = env.reset(seed=69)[0]
        truncated = False
        terminated = False
        done = False
        while not done:
            if (rng.uniform(0, 1) < epsilon) or (np.all(Q[state, :]) == Q[state, 0]):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
        
            new_state, reward, terminated, truncated, info = env.step(action)
            done  = terminated or truncated
            all_states.append(new_state)
            all_actions.append(action)
            expected_future_reward = np.max(Q[new_state, :]) 
            Q[state, action] += learningRate * (reward + discountRate * expected_future_reward - Q[state, action])

            state = new_state
        if reward == 1:
            rewards_per_episode[episode] = 1
    return Q, rewards_per_episode, all_states, all_actions

def plot_states_actions_distribution(states, actions, map_size):
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(img_title)
    # plt.show()

def plot_cum_rewards(rewards_per_episode, episodes):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-1000):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lakeq4x4.png')

def qtable_directions_map(qtable, map_size):
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    fig.suptitle(f"Optimal policy and Q-values heatmap(Grid Size: {map_size}x{map_size})")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(img_title)
    plt.show()


def run_policy(Q_table, slippery, map_name='4x4',desc=None):
    if desc:
        env = gym.make('FrozenLake-v1', is_slippery=slippery, desc=desc, render_mode='human')
    else:
        env = gym.make('FrozenLake-v1', is_slippery=slippery, map_name=map_name, render_mode="human")
    state = env.reset()[0]
    done = False
    while not done:
      env.render()
      action = np.argmax(Q_table[state])
      state, reward, terminated, truncated, _= env.step(action)
      done = terminated or truncated
      if reward == 1:
          env.close()
          return 1
    env.close()
    return 0


# env = gym.make('FrozenLake-v1', is_slippery=True, map_name='4x4', render_mode="rgb_array")

# Q_table, rewards_per_episode, states, actions = q_learning(env)
# plot_cum_rewards(rewards_per_episode, EPISODES)
# plot_states_actions_distribution(states, actions, map_size=4)
# plot_q_values_map(Q_table, env, 4)
# env.close()
# print(Q_table)

# env = gym.make('FrozenLake-v1', is_slippery=True, map_name='4x4', render_mode="human")
# state = env.reset()[0]
# done = False
# while not done:
#   env.render()
#   action = np.argmax(Q_table[state])
#   state, _, terminated, truncated, _= env.step(action)
#   done = terminated or truncated
# env.close()