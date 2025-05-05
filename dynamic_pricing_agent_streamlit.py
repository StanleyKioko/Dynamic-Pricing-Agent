import numpy as np
import gym
from gym import spaces
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Custom CSS for e-commerce styling
st.markdown("""
<style>
    .product-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .product-image {
        max-width: 100%;
        border-radius: 8px;
    }
    .product-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .product-description {
        color: #666;
        margin: 10px 0;
    }
    .product-price {
        font-size: 20px;
        font-weight: bold;
        color: #2ecc71;
    }
    .simulate-button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .simulate-button:hover {
        background-color: #2980b9;
    }
    .dashboard-header {
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Custom Gym Environment for Dynamic Pricing
class PricingEnv(gym.Env):
    def __init__(self):
        super(PricingEnv, self).__init__()
        self.inventory_levels = 101  # 0 to 100 units
        self.demand_levels = 3  # Low (0), Medium (1), High (2)
        self.action_space = spaces.Discrete(3)  # Prices: $500 (0), $750 (1), $1000 (2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.inventory_levels),
            spaces.Discrete(self.demand_levels)
        ))
        self.cost = 400  # Cost per unit
        self.prices = [500, 750, 1000]  # Realistic smartphone prices
        self.max_days = 30  # Episode length
        self.base_demand = 20  # Base demand
        self.demand_factor = 0.01  # Demand decreases with price
        self.reset()

    def reset(self):
        self.inventory = 100
        self.day = 0
        self.demand_level = 1
        return (self.inventory, self.demand_level)

    def step(self, action):
        price = self.prices[action]
        demand = max(0, int(self.base_demand - self.demand_factor * price))
        if demand > 15:
            self.demand_level = 2
        elif demand > 5:
            self.demand_level = 1
        else:
            self.demand_level = 0
        units_sold = min(demand, self.inventory)
        self.inventory -= units_sold
        reward = (price - self.cost) * units_sold
        self.day += 1
        done = self.inventory == 0 or self.day >= self.max_days
        state = (self.inventory, self.demand_level)
        info = {"price": price, "units_sold": units_sold, "profit": reward}
        return state, reward, done, info

    def render(self):
        pass

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.q_table = np.zeros((env.inventory_levels, env.demand_levels, env.action_space.n))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        next_max_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state[0], state[1], action] = new_q

# Training Function
def train_agent(episodes, learning_rate, discount_factor, epsilon):
    env = PricingEnv()
    agent = QLearningAgent(env)
    agent.lr = learning_rate
    agent.gamma = discount_factor
    agent.epsilon = epsilon
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
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        progress = (episode + 1) / episodes
        progress_bar.progress(progress)
        status_text.text(f"Training: Episode {episode + 1}/{episodes}, Profit: ${episode_reward:.2f}")

    return total_rewards, prices_over_time, profits_over_time

# Streamlit Interface
st.title("Dynamic Pricing E-Commerce Demo")
st.markdown("Welcome to our AI-powered e-commerce store! Watch the price of the Galaxy Z Fold adjust dynamically based on demand and inventory, optimized by a reinforcement learning agent.")

# Product Section
st.markdown("<div class='product-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://images.unsplash.com/photo-1598327105666-5b89351aff97", caption="Galaxy Z Fold", use_column_width=True, output_format="auto")
with col2:
    st.markdown("<h2 class='product-title'>Galaxy Z Fold</h2>", unsafe_allow_html=True)
    st.markdown("<p class='product-description'>Experience the future with the Galaxy Z Fold, featuring a foldable display and cutting-edge technology.</p>", unsafe_allow_html=True)
    if 'last_prices' not in st.session_state:
        st.session_state.last_prices = [750]  # Initial price
        st.session_state.current_day = 0
    st.markdown(f"<p class='product-price'>Price: ${st.session_state.last_prices[st.session_state.current_day]}</p>", unsafe_allow_html=True)
    if st.button("Simulate Next Day", key="simulate"):
        if st.session_state.current_day < len(st.session_state.last_prices) - 1:
            st.session_state.current_day += 1
        else:
            st.warning("No more days to simulate. Please train the agent again.")
st.markdown("</div>", unsafe_allow_html=True)

# Dashboard Section
st.markdown("<h2 class='dashboard-header'>Pricing Agent Dashboard</h2>", unsafe_allow_html=True)

# Parameter Inputs
st.sidebar.header("Training Parameters")
episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
discount_factor = st.sidebar.slider("Discount Factor", 0.1, 0.99, 0.9, step=0.01)
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.01, 0.5, 0.1, step=0.01)

# Train Button
if st.button("Train Agent"):
    with st.spinner("Training agent..."):
        total_rewards, prices_over_time, profits_over_time = train_agent(
            episodes, learning_rate, discount_factor, epsilon
        )
        st.session_state.last_prices = prices_over_time[-1]
        st.session_state.current_day = 0

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