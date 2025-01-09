import numpy as np
import matplotlib.pyplot as plt
import random

# Bandit class
class Bandit:
    def __init__(self, mean=0, stddev=1):
        self.__mean = mean
        self.__stddev = stddev
        self.estimated_value = 0
        self.pull_count = 0

    '''Simulates pulling the lever of the bandit and returns the reward'''
    def pullLever(self):
        return np.random.normal(self.__mean, self.__stddev)

    def update_estimated_value(self, reward):
        self.pull_count += 1
        self.estimated_value += (reward - self.estimated_value) / self.pull_count

# Initialization
bandits = [Bandit(random.random() * 4 - 2) for _ in range(10)]

# Greedy Algorithm
def run_greedy(iterations=1000):
    rewards = []
    for _ in range(iterations):
        # Choose the bandit with the highest estimated value
        best_bandit = max(bandits, key=lambda b: b.estimated_value)
        reward = best_bandit.pullLever()
        best_bandit.update_estimated_value(reward)
        rewards.append(reward)
    return rewards

# Epsilon-Greedy Algorithm
def run_epsilon_greedy(epsilon, iterations=1000):
    rewards = []
    for _ in range(iterations):
        if random.random() < epsilon:
            # Explore: Choose a random bandit
            bandit = random.choice(bandits)
        else:
            # Exploit: Choose the best bandit
            bandit = max(bandits, key=lambda b: b.estimated_value)
        reward = bandit.pullLever()
        bandit.update_estimated_value(reward)
        rewards.append(reward)
    return rewards

# Optimistic Initial Values
def run_optimistic_greedy(initial_value, iterations=1000):
    for bandit in bandits:
        bandit.estimated_value = initial_value
    rewards = []
    for _ in range(iterations):
        best_bandit = max(bandits, key=lambda b: b.estimated_value)
        reward = best_bandit.pullLever()
        best_bandit.update_estimated_value(reward)
        rewards.append(reward)
    return rewards

# Upper Confidence Bound (UCB)
def run_ucb(c, iterations=1000):
    rewards = []
    total_pulls = 0
    for _ in range(iterations):
        ucb_values = [
            bandit.estimated_value + c * np.sqrt(np.log(total_pulls + 1) / (bandit.pull_count + 1e-5))
            for bandit in bandits
        ]
        best_bandit = bandits[np.argmax(ucb_values)]
        reward = best_bandit.pullLever()
        best_bandit.update_estimated_value(reward)
        rewards.append(reward)
        total_pulls += 1
    return rewards

# Plot cumulative average rewards
def plot_rewards(rewards, title):
    cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.plot(cumulative_avg, label=title)
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()

# Main execution
if __name__ == "__main__":
    plt.figure(figsize=(12, 8))

    # Run and plot Greedy
    greedy_rewards = run_greedy()
    plot_rewards(greedy_rewards, "Greedy")

    # Run and plot Epsilon-Greedy for various epsilons
    for epsilon in [0.1, 0.2, 0.5]:
        for bandit in bandits:
            bandit.estimated_value = 0
            bandit.pull_count = 0
        epsilon_rewards = run_epsilon_greedy(epsilon)
        plot_rewards(epsilon_rewards, f"Epsilon-Greedy (\u03b5={epsilon})")

    # Run and plot Optimistic Greedy
    for bandit in bandits:
        bandit.estimated_value = 0
        bandit.pull_count = 0
    optimistic_rewards = run_optimistic_greedy(initial_value=1)
    plot_rewards(optimistic_rewards, "Optimistic Greedy")

    # Run and plot UCB
    for bandit in bandits:
        bandit.estimated_value = 0
        bandit.pull_count = 0
    ucb_rewards = run_ucb(c=2)
    plot_rewards(ucb_rewards, "UCB (c=2)")

    plt.title("Cumulative Average Rewards")
    plt.show()
