# This code implements a simple policy gradient algorithm (REINFORCE) to solve the CartPole-v1 environment from OpenAI's gym.
# Unfortunately, it does not work as expected. The agent does not learn to balance the pole.
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
    """
    Compute the discounted returns for a list of rewards.

    "Returns" means the total reward received after taking an action, discounted by gamma.
    """
    R = 0
    returns = []
    # Reverse the rewards list to compute returns from the end of the episode
    for r in reversed(rewards):
        R = r + gamma * R
        # This insert(0, R) negates the reverse order of the rewards to the original order
        returns.insert(0, R)
    return returns


# Training loop
def train(env, policy, optimizer, episodes=1000, gamma=0.99):
    all_rewards = []

    # Train for a number of episodes. Each episode is a complete run of the environment until either the agent wins or loses.
    for episode in range(episodes):
        # Reset the environment and initialize the state with certain randomness
        state, _ = env.reset()

        # Store the log probabilities of actions taken
        log_probs = []

        # Store the rewards for this episode
        rewards = []

        # Flag to indicate if the episode is done
        done = False
        while not done:

            # Convert the state of shape (4,) to a tensor of shape (1, 4)
            # (4,) means 4 elements in a single dimension
            # (1, 4) means 1 row and 4 columns
            # This is necessary because the neural network expects a batch of inputs
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get the action probabilities from the policy network
            probs = policy(state_tensor)

            # Sample an action from the categorical distribution
            dist = Categorical(probs)
            action = dist.sample()

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # Store the log probability of the action taken
            log_probs.append(dist.log_prob(action))

            # Store the reward received
            rewards.append(reward)

            # Update the state
            state = next_state

            # Check if the episode is done
            done = terminated or truncated

        # Compute the "raw" returns
        returns = compute_returns(rewards, gamma)

        # Normalize the returns. Why? Because the scale of the returns can vary significantly, and normalizing them helps stabilize training.
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute the loss. The loss is the negative log probability of the actions taken, weighted by the returns.
        # Why? Because we want to maximize the expected return, which is equivalent to minimizing the negative log probability of the actions taken.
        # So that after backpropagation, the policy network will learn to make the probabilities of the actions resulting in higher returns larger.
        # Note: torch.stack turns the list of 1x1 tensors into a single tensor of shape (N, 1), where N is the number of actions taken in the episode.
        loss = -torch.sum(torch.stack(log_probs) * returns)

        # zero_grad() clears old gradients from the last step in the optimizer (otherwise they will accumulate).
        optimizer.zero_grad()

        # Backpropagation: compute the gradients of the loss with respect to the parameters of the policy network.
        # The resulting gradients will be stored in the .grad attribute of the parameters of the policy network.
        loss.backward()

        # Step the optimizer to update the parameters of the policy network.
        optimizer.step()

        # Sum the rewards for this episode and append to the list of all rewards only for logging and plotting purposes
        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, reward: {episode_reward:.2f}")

    return all_rewards


# Main execution
if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # env.observation_space shows the range of values for each state
    state_dim = env.observation_space.shape[0]
    print("State space dimension:", state_dim)

    # env.action_space shows the number of possible actions
    action_dim = env.action_space.n
    print("Action space dimension:", action_dim)

    # Define a neural network as policy, providing the action probabilities with softmax given the state
    policy = PolicyNet(state_dim, action_dim)

    # Set upthe optimizer
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    # Train the policy using the REINFORCE algorithm
    rewards = train(env, policy, optimizer, episodes=2000)

    # Plot reward over time
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Policy Gradient on CartPole")
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
        time.sleep(0.2)
    env.close()
