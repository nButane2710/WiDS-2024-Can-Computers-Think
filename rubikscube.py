import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Rubik's Cube environment with actual mechanics
class RubiksCubeEnv:
    def __init__(self):
        self.reset()
        self.action_space = 12  # 6 faces * 2 directions

    def reset(self):
        self.state = np.array([i // 9 for i in range(54)])  # Solved state
        for _ in range(20):  # Apply random moves to scramble
            self.state = self._apply_move(self.state, random.randint(0, 11))
        return self.state.copy()

    def step(self, action):
        next_state = self._apply_move(self.state, action)
        reward = 1 if self.is_solved(next_state) else -0.1
        done = self.is_solved(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def is_solved(self, state):
        return all(np.all(state[i*9:(i+1)*9] == state[i*9]) for i in range(6))

    def _apply_move(self, state, move):
        # Moves: 0-5 (clockwise), 6-11 (counterclockwise)
        face = move % 6
        clockwise = move < 6
        return self._rotate_face(state, face, clockwise)

    def _rotate_face(self, state, face, clockwise=True):
        # Rotation mappings for Rubik's Cube faces
        face_indices = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],    # Up
            [9, 10, 11, 12, 13, 14, 15, 16, 17],  # Left
            [18, 19, 20, 21, 22, 23, 24, 25, 26], # Front
            [27, 28, 29, 30, 31, 32, 33, 34, 35], # Right
            [36, 37, 38, 39, 40, 41, 42, 43, 44], # Back
            [45, 46, 47, 48, 49, 50, 51, 52, 53]  # Down
        ]
        # Simplified face rotation (just rotates face for now)
        idx = face_indices[face]
        if clockwise:
            rotated = np.array([state[idx[i]] for i in [6, 3, 0, 7, 4, 1, 8, 5, 2]])
        else:
            rotated = np.array([state[idx[i]] for i in [2, 5, 8, 1, 4, 7, 0, 3, 6]])
        new_state = state.copy()
        new_state[idx] = rotated
        # TODO: Add adjacent face rotation logic
        return new_state

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop
env = RubiksCubeEnv()
state_size = 54
action_size = env.action_space
agent = DQNAgent(state_size, action_size)
batch_size = 32

for e in range(1000):
    state = env.reset()
    done = False
    time = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        time += 1
        if done:
            agent.update_target_model()
            print(f"Episode {e+1}: {time} steps")
    agent.replay(batch_size)

# Evaluation
test_state = env.reset()
done = False
steps = 0
while not done:
    action = agent.act(test_state)
    test_state, _, done, _ = env.step(action)
    steps += 1

print(f"Solved in {steps} steps")
