# This code implements the advantage actor-critic (A2C) algorithm to solve the CartPole-v1 environment from OpenAI's gym.
# It works well but seems to be slightly less efficient than the REINFORCE with baseline implementation, at least for the CartPole-v1 environment.
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


# Training function
def train(
    env,
    policy,
    value_net,
    policy_optimizer,
    value_optimizer,
    episodes=1000,
    gamma=0.99,
    batch_size=10,
    entropy_beta=0.01,
):
    # Store total rewards for each episode
    episode_rewards = []
    for episode in range(episodes):
        # Reset the environment and initialize the state with certain randomness
        state, _ = env.reset()

        # Reset variables
        states, log_probs, values, advantages, td_targets = [], [], [], [], []
        episode_reward = 0

        # Flag to indicate if the episode is done
        done = False
        while not done:
            # Convert state to tensor and get action probabilities
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)

            # Sample action from the policy
            dist = Categorical(probs)
            action = dist.sample()

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # Check if the episode is done
            done = terminated or truncated

            # Convert next state to tensor
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Compute TD target and advantage
            # TD means temporal difference. This TD target is the TD(0) target, i.e., the
            # one step ahead target. When the entire episode is used, TD(n)
            # converges to the Monte Carlo learning approach.
            value = value_net(state_tensor)
            next_value = (
                value_net(next_state_tensor) if not done else torch.tensor([0.0])
            )
            td_target = reward + gamma * next_value.detach()
            advantage = td_target - value

            # Store training data
            states.append(state)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            advantages.append(advantage)
            td_targets.append(td_target)

            episode_reward += reward
            state = next_state

        log_probs_tensor = torch.stack(log_probs).squeeze()
        advantages_tensor = torch.stack(advantages).squeeze().detach()

        # Policy loss with entropy bonus
        entropy = -torch.sum(
            torch.stack(
                [
                    (p := policy(torch.FloatTensor(s).unsqueeze(0))).squeeze(0)
                    * p.log().squeeze(0)
                    for s in states
                ]
            )
        ) / len(states)
        policy_loss = (
            -(log_probs_tensor * advantages_tensor).mean() - entropy_beta * entropy
        )

        # Value loss
        td_targets_tensor = torch.stack(td_targets).squeeze().detach()
        values_tensor = torch.stack(values).squeeze()
        value_loss = nn.functional.mse_loss(values_tensor, td_targets_tensor)

        # Optimize policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()

        # Optimize value network
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % batch_size == 0:
            avg = np.mean(episode_rewards[-batch_size:])
            print(f"Episode {episode+1}, avg reward (last {batch_size}): {avg:.2f}")

    return episode_rewards


# Run the training
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    # Get the dimensions of the state and action spaces
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the policy network, which takes the
    # state as input and outputs the action probabilities.
    policy = PolicyNet(state_dim, action_dim)

    # Initialize the value network, which
    # takes the state as input and outputs the value of that state, which is
    # the expected return from that state.
    value_net = ValueNet(state_dim)

    # Set up the optimizers for both networks
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    # Train the policy using the actor-critic method
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
