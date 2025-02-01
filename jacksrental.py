import numpy as np
from scipy.stats import poisson

# Constants
MAX_CARS = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9
THRESHOLD = 1e-4

# Poisson parameters
RENTAL_REQUEST_FIRST = 3
RENTAL_REQUEST_SECOND = 4
RETURNS_FIRST = 3
RETURNS_SECOND = 2

# Precompute Poisson probabilities
poisson_cache = dict()
def poisson_prob(n, lam):
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

# Initialize value function and policy
V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
policy = np.zeros(V.shape, dtype=int)

# Expected return calculation
def expected_return(state, action, V):
    returns = -MOVE_COST * abs(action)
    cars_first_loc = min(state[0] - action, MAX_CARS)
    cars_second_loc = min(state[1] + action, MAX_CARS)

    for rental_first in range(0, 11):
        for rental_second in range(0, 11):
            prob_rental = (
                poisson_prob(rental_first, RENTAL_REQUEST_FIRST) *
                poisson_prob(rental_second, RENTAL_REQUEST_SECOND)
            )

            real_rental_first = min(cars_first_loc, rental_first)
            real_rental_second = min(cars_second_loc, rental_second)

            reward = (real_rental_first + real_rental_second) * RENTAL_REWARD

            cars_first_loc_ = cars_first_loc - real_rental_first
            cars_second_loc_ = cars_second_loc - real_rental_second

            for return_first in range(0, 11):
                for return_second in range(0, 11):
                    prob_return = (
                        poisson_prob(return_first, RETURNS_FIRST) *
                        poisson_prob(return_second, RETURNS_SECOND)
                    )

                    final_first = min(cars_first_loc_ + return_first, MAX_CARS)
                    final_second = min(cars_second_loc_ + return_second, MAX_CARS)

                    prob = prob_rental * prob_return
                    returns += prob * (reward + DISCOUNT * V[final_first, final_second])

    return returns

# Policy Iteration
policy_stable = False
while not policy_stable:
    # Policy Evaluation
    while True:
        delta = 0
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                v = V[i, j]
                V[i, j] = expected_return((i, j), policy[i, j], V)
                delta = max(delta, abs(v - V[i, j]))
        if delta < THRESHOLD:
            break

    # Policy Improvement
    policy_stable = True
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            old_action = policy[i, j]
            action_returns = []
            for action in range(-MAX_MOVE, MAX_MOVE + 1):
                if (0 <= i - action <= MAX_CARS) and (0 <= j + action <= MAX_CARS):
                    action_returns.append(expected_return((i, j), action, V))
                else:
                    action_returns.append(-np.inf)

            best_action = np.argmax(action_returns) - MAX_MOVE
            policy[i, j] = best_action

            if old_action != best_action:
                policy_stable = False

# Save the policy and value function
np.save('optimal_policy.npy', policy)
np.save('optimal_value.npy', V)

print("Optimal policy and value function saved.")
