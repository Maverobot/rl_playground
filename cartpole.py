import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


# Function to compute discounted rewards
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# Training loop
def train(env, policy, optimizer, episodes=1000, gamma=0.99):
    all_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            done = terminated or truncated

        # Compute returns and loss
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, reward: {episode_reward:.2f}")

    return all_rewards


# Main execution
if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    rewards = train(env, policy, optimizer, episodes=2000)

    # Plot reward over time
    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("Policy Gradient on CartPole")
    # plt.show()

    # Show final policy
    env = gym.make("CartPole-v1", render_mode="human")
    while True:
        env.reset()
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            action = probs.argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
        time.sleep(0.2)
    env.close()
    # Note: Make sure to install the required libraries before running the code.
