import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def value_iteration(env, discountFactor=0.99, theta=1e-6):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    pi = np.zeros(env.observation_space.n, dtype=int)
    V = np.zeros(state_size)
    change_per_it =[]
    iterations = 0
    while True:
        delta = 0
        for state in range(state_size):
            v = V[state]
            v_new = np.zeros(action_size)
            for action in range(action_size):
                for prob, next_state, reward, terminal in env.get_wrapper_attr('P')[state][action]:
                    v_new[action] += prob * (reward + discountFactor * V[next_state])
            V[state] = np.max(v_new)
            pi[state] = np.argmax(v_new)
            delta = max(delta, np.abs(v - V[state]))
        iterations += 1
        change_per_it.append(delta)
        if delta < theta:
            break
    return V, iterations, change_per_it, pi

def plot_value_function(iterations, change_per_it, filename=None):
    iterations = range(len(change_per_it))
    plt.plot(iterations, change_per_it, label='Value Change')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Change in Value')
    plt.title('Convergence of State Value Function in FrozenLake-v1')
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def get_directions(optimal_V, policy, reshapeDim):
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    best_actions = optimal_V.reshape(reshapeDim, reshapeDim)
    directions_table = np.empty(best_actions.flatten().shape, dtype=str)
    for idx, val in enumerate(best_actions.flatten()):
        if best_actions.flatten()[idx] > 0:
            directions_table[idx] = directions[policy[idx]]
    directions_table = directions_table.reshape(reshapeDim, reshapeDim)
    return best_actions, directions_table

def grid_print(optimal_V, policy, reshapeDim):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    best_actions, direction_table = get_directions(optimal_V, policy, reshapeDim)
    sns.heatmap(best_actions, ax=ax1, 
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    
    ax1.set_title("Value Function")
    sns.heatmap(best_actions, ax=ax2, annot=direction_table, square=True, cbar=False, fmt="", cmap='Blues', xticklabels=False, yticklabels=False, linewidths=0.7,linecolor="black",)
    ax2.set_title("Optimal Actions (Arrows)")
 
    fig.suptitle(f"Value Function and Optimal Policy (Grid Size: {reshapeDim}x{reshapeDim})")
    plt.savefig('valueFunctionGrid.png',dpi=600)
    plt.show()

def run_policy(slippery, policy, map_name='4x4', desc=None):
    if desc:
        env = gym.make('FrozenLake-v1', is_slippery=slippery, desc=desc, render_mode='human')
    else:
        env = gym.make('FrozenLake-v1', is_slippery=slippery, map_name=map_name, render_mode='human')
    state = env.reset()[0]
    done = False
    while not done:
        env.render()
        action = policy[int(state)]
        state, reward, terminated, truncated, _= env.step(action)
        done = terminated or truncated
        if reward == 1:
            env.close()
            return 1
    env.close()
    return 0



# env = gym.make('FrozenLake-v1', is_slippery=True, map_name='4x4', render_mode='human')
# observation, info = env.reset(seed=69)
# optimal_V, iterations, change_per_it, policy = value_iteration(env)

# plot_value_function(iterations, change_per_it, filename='value_iteration_convergence.png')
# grid_print(optimal_V, policy, 4)

