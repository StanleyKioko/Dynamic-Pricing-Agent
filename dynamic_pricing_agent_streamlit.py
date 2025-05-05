import numpy as np
import gym
from gym import spaces
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Custom Gym Environment for Dynamic Pricing
class PricingEnv(gym.Env):
    def __init__(self):
        super(PricingEnv, self).__init__()
        # State: (inventory_level, demand_level)
        self.inventory_levels = 101  # 0 to 100 units
        self.demand_levels = 3  # Low (0), Medium (1), High (2)
        self.action_space = spaces.Discrete(3)  # Prices: $50 (0), $75 (1), $100 (2)
        self.observation_space  # Prices: $50 (0), $75 (1), $100 (2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.inventory_levels),
            spaces.Discrete(self.demand_levels)
        ))
        # Parameters
        self.cost = 40  # Cost per unit
        self.prices = [50, 75, 100]  # Available prices
        self.max_days = 30  # Episode length
        self.base_demand = 20  # Base demand (adjusted by price)
        self.demand_factor = 0.1  # Demand decreases with price
        self.reset()

    def reset(self):
        self.inventory = 100  # Start with 100 units
        self.day = 0
        self.demand_level = 1  # Start with medium demand
        return (self.inventory, self.demand_level)

    def step(self, action):
        price = self.prices[action]
        # Calculate demand based on price
        demand = max(0, int(self.base_demand - self.demand_factor * price))
        # Demand level: High (demand > 15), Medium (5-15), Low (< 5)
        if demand > 15:
            self.demand_level = 2
        elif demand > 5:
            self.demand_level = 1
        else:
            self.demand_level = 0
        # Sales limited by inventory
        units_sold = min(demand, self.inventory)
        self.inventory -= units_sold
        # Calculate reward (profit)
        reward = (price - self.cost) * units_sold
        self.day += 1
        # Done if inventory is 0 or max days reached
        done = self.inventory == 0 or self.day >= self.max_days
        # Next state
        state = (self.inventory, self.demand_level)
        info = {"price": price, "units_sold": units_sold, "profit": reward}
        return state, reward, done, info

    def render(self):
        pass  # No rendering needed for Streamlit

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Initialize Q-table
        self.q_table = np.zeros((env.inventory_levels, env.demand_levels, env.action_space.n))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:  # Exploration
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state[0], state[1]])  # Exploitation

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        next_max_q = np.max(self.q_table[next_state[0], next_state[1]])
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state[0], state[1], action] = new_q

# Training Function
def train_agent(episodes, learning_rate, discount_factor, epsilon):
    env = PricingEnv()
    agent = QLearningAgent(env, learning_rate, discount_factor, epsilon)
    total_rewards = []
    prices_over_time = []
    profits_over_time = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_prices = []
        episode_profits = []

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            episode_prices.append(info["price"])
            episode_profits.append(info["profit"])

            if done:
                break

        total_rewards.append(episode_reward)
        prices_over_time.append(episode_prices)
        profits_over_time.append(episode_profits)
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # Update progress
        progress = (episode + 1) / episodes
        progress_bar.progress(progress)
        status_text.text(f"Training: Episode {episode + 1}/{episodes}, Profit: ${episode_reward:.2f}")

    return total_rewards, prices_over_time, profits_over_time, agent

# Streamlit Interface
st.title("Dynamic Pricing Agent Dashboard")
st.markdown("""
This dashboard demonstrates an AI agent that uses Q-learning to optimize product pricing in a simulated retail market.
Adjust the parameters below and click 'Train Agent' to see the agent learn!
""")

# Parameter Inputs
st.sidebar.header("Training Parameters")
episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
discount_factor = st.sidebar.slider("Discount Factor", 0.1, 0.99, 0.9, step=0.01)
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.01, 0.5, 0.1, step=0.01)

# Train Button
if st.button("Train Agent"):
    with st.spinner("Training agent..."):
        total_rewards, prices_over_time, profits_over_time, agent = train_agent(
            episodes, learning_rate, discount_factor, epsilon
        )

    # Display Results
    st.header("Training Results")

    # Plot Total Rewards
    st.subheader("Total Profit per Episode")
    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Profit ($)")
    ax.set_title("Learning Curve")
    st.pyplot(fig)

    # Plot Prices and Profits in Last Episode
    st.subheader("Pricing and Profits in Last Episode")
    last_prices = prices_over_time[-1]
    last_profits = profits_over_time[-1]
    df = pd.DataFrame({
        "Day": range(1, len(last_prices) + 1),
        "Price ($)": last_prices,
        "Profit ($)": last_profits
    })

    # Line Plot
    fig, ax = plt.subplots()
    ax.plot(df["Day"], df["Price ($)"], label="Price", color="blue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price ($)")
    ax2 = ax.twinx()
    ax2.plot(df["Day"], df["Profit ($)"], label="Profit", color="green")
    ax2.set_ylabel("Profit ($)")
    ax.set_title("Prices and Profits")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

    # Table of Results
    st.subheader("Summary Metrics")
    st.write(f"**Average Profit per Episode**: ${np.mean(total_rewards):.2f}")
    st.write(f"**Max Profit Achieved**: ${np.max(total_rewards):.2f}")
    st.write(f"**Final Episode Profit**: ${total_rewards[-1]:.2f}")
    st.dataframe(df)