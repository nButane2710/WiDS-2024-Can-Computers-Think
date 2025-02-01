import numpy as np
import matplotlib.pyplot as plt

def gambler_problem(p_h=0.4, theta=1e-9, max_capital=100):
    """
    Solves the Gambler's Problem using Value Iteration.
    :param p_h: Probability of winning the bet (default 0.4)
    :param theta: Convergence threshold
    :param max_capital: Maximum capital (goal state)
    :return: Optimal state values and policy
    """
    # State values (V) and policy (π)
    V = np.zeros(max_capital + 1)
    policy = np.zeros(max_capital + 1)
    
    while True:
        delta = 0
        for s in range(1, max_capital):  # Capital states (excluding 0 and 100)
            actions = range(1, min(s, max_capital - s) + 1)  # Possible bets
            action_returns = []
            
            for a in actions:
                win = p_h * V[s + a]  # Win case
                lose = (1 - p_h) * V[s - a]  # Lose case
                action_returns.append(win + lose)
            
            max_return = max(action_returns)
            delta = max(delta, abs(max_return - V[s]))
            V[s] = max_return
        
        if delta < theta:
            break
    
    # Deriving policy from optimal values
    for s in range(1, max_capital):
        actions = range(1, min(s, max_capital - s) + 1)
        action_returns = [(p_h * V[s + a] + (1 - p_h) * V[s - a]) for a in actions]
        policy[s] = actions[np.argmax(action_returns)]  # Best action
    
    return V, policy

def plot_results(V, policy):
    """Plots the optimal state values and policy."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].plot(V)
    ax[0].set_title("Optimal State Values")
    ax[0].set_xlabel("Capital")
    ax[0].set_ylabel("Value")
    
    ax[1].step(range(len(policy)), policy, where='mid')
    ax[1].set_title("Optimal Policy")
    ax[1].set_xlabel("Capital")
    ax[1].set_ylabel("Stake")
    
    plt.show()

if __name__ == "__main__":
    V, policy = gambler_problem()
    plot_results(V, policy)
