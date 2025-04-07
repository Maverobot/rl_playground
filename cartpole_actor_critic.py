import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Actor network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


# Critic network (baseline)
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# Compute discounted rewards
def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# Training function
def train(
    env,
    policy,
    value_fn,
    policy_optimizer,
    value_optimizer,
    episodes=1000,
    gamma=0.99,
    batch_size=10,
    entropy_beta=0.01,
):
    all_rewards = []
    episode_rewards = []

    for episode in range(episodes):
        states, actions, rewards, log_probs = [], [], [], []

        state, _ = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Add reward shaping to encourage keeping the cart centered
            reward += (4.8 - abs(next_state[0])) / 4.8 * 1

            # Save experience
            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        # Compute returns and advantages
        returns = compute_returns(rewards, gamma)
        states_tensor = torch.FloatTensor(states)
        returns_tensor = torch.FloatTensor(returns)
        values = value_fn(states_tensor)
        advantages = returns_tensor - values.detach()

        # Policy loss
        log_probs_tensor = torch.stack(log_probs)
        entropy = -torch.sum(
            torch.stack(
                [
                    (p := policy(torch.FloatTensor(s).unsqueeze(0))).squeeze(0)
                    * p.log().squeeze(0)
                    for s in states
                ]
            )
        ) / len(states)

        policy_loss = -(log_probs_tensor * advantages).mean() - entropy_beta * entropy

        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(values, returns_tensor)

        # Optimize policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()

        # Optimize value network
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        if (episode + 1) % batch_size == 0:
            avg = np.mean(all_rewards[-batch_size:])
            print(f"Episode {episode+1}, avg reward (last {batch_size}): {avg:.2f}")

        episode_rewards.append(episode_reward)

    return episode_rewards


# Run the training
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)

    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    rewards = train(
        env, policy, value_net, policy_optimizer, value_optimizer, episodes=1500
    )

    # Plot results
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Improved REINFORCE (Actor-Critic) on CartPole")
    plt.show(block=False)

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
            if done:
                print(
                    f"Episode finished after {len(rewards)} timesteps. Terminated: {terminated}, truncated: {truncated}"
                )
        time.sleep(0.2)
    env.close()
