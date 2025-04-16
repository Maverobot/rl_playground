# This code implements REINFORCE with a baseline (value function) to reduce variance in the policy gradient estimates.
# It has been tested in following environments:
# - CartPole-v1: it works well with 1500 episodes
# - LunarLander-v3: the lander is able to land properly after about 7000 episodes but it does not yet learn to turn off the engines once landed.
import argparse
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
    value_net,
    policy_optimizer,
    value_optimizer,
    episodes=1000,
    gamma=0.99,
    batch_size=10,
    entropy_beta=0.01,
    preemption_reward=None,
):
    # Store total rewards for each episode
    episode_rewards = []
    for episode in range(episodes):
        # Reset variables
        states, actions, rewards, log_probs = [], [], [], []

        # Reset the environment and initialize the state with certain randomness
        state, _ = env.reset()

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

            # Mark episode as done if terminated or truncated
            done = terminated or truncated

            # Save experience
            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        # Compute the "raw" returns
        returns = compute_returns(rewards, gamma)
        returns_tensor = torch.FloatTensor(returns)

        # Compute the values from the value network for each state in the episode
        states_tensor = torch.FloatTensor(states)
        values = value_net(states_tensor)

        # Policy loss. It is used to train policy network by increasing the
        # adantage over the baseline i.e. the value function.
        advantages = returns_tensor - values.detach()
        # TODO(qu): Understand the entropy term and why it is added
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

        # Value loss (MSE). It is used to train the value network to predict
        # the returns.
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
        episode_rewards.append(episode_reward)

        if (episode + 1) % batch_size == 0:
            avg = np.mean(episode_rewards[-batch_size:])
            print(f"Episode {episode+1}, avg reward (last {batch_size}): {avg:.2f}")
            if preemption_reward is not None and avg >= preemption_reward:
                print(
                    f"Preemption reward threshold of {preemption_reward} reached. Stopping training."
                )
                break

    return episode_rewards


# Run the training
if __name__ == "__main__":
    # Get environment name from command line argument
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Environment name"
    )
    argument_parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    argument_parser.add_argument(
        "--training-preemption-reward",
        type=float,
        default=None,
        help="Preemption reward threshold of the training episode",
    )
    args = argument_parser.parse_args()
    env_name = args.env

    env = gym.make(env_name, render_mode="rgb_array")

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
        env,
        policy,
        value_net,
        policy_optimizer,
        value_optimizer,
        episodes=args.episodes,
        preemption_reward=args.training_preemption_reward,
    )

    # Plot results
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Improved REINFORCE (Actor-Critic) on {env_name}")
    plt.show(block=False)

    # Show final policy
    env = gym.make(env_name, render_mode="human")
    while True:
        env.reset()
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            action = probs.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated
            if done:
                print(
                    f"Episode finished with reward {reward}. Terminated: {terminated}, truncated: {truncated}"
                )
        time.sleep(0.2)
    env.close()
