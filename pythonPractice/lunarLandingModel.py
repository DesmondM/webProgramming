import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
import imageio

# Hyperparameters
EPISODES = 1000
MAX_STEPS = 1000
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MEMORY_SIZE = 100_000
BATCH_SIZE = 64
TARGET_UPDATE = 10
VIDEO_FILENAME = "lunarlander_dqn.mp4"

env = gym.make('LunarLander-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Epsilon-greedy action selection
def select_action(state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return torch.argmax(q_values).item()

# Training step
def train():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize everything
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = QNetwork(state_size, action_size).to(device)
target_net = QNetwork(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

eps = EPS_START

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS):
        action = select_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()
        if done:
            break

    eps = max(EPS_END, eps * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {eps:.3f}")

    if total_reward >= 200:
        print(f"\n🎉 Environment solved in episode {episode}!\n")
        break

# Save trained model
torch.save(policy_net.state_dict(), "dqn_lunarlander.pth")

# Generate a video of the agent
frames = []
state = env.reset()
done = False
while not done:
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.argmax(policy_net(state_tensor)).item()
    state, _, done, _ = env.step(action)

env.close()

# Write video
imageio.mimsave(VIDEO_FILENAME, frames, fps=30)
print(f"\n📹 Video saved as {VIDEO_FILENAME}")
