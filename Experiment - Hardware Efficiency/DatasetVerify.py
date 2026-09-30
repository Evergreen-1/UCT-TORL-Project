import minari
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Minari dataset
dataset_id = "mujoco/halfcheetah/medium-v0"  # Adjust exact dataset ID if using custom/different version
dataset = minari.load_dataset(dataset_id, download=True)

# 2. Extract total episode returns and lengths
returns = []
episode_lengths = []

for episode in dataset.iterate_episodes():
    ep_return = np.sum(episode.rewards)
    returns.append(ep_return)
    episode_lengths.append(len(episode.rewards))

returns = np.array(returns)
episode_lengths = np.array(episode_lengths)

# 3. Print Summary Statistics
print(f"=== Dataset Return Summary ({dataset_id}) ===")
print(f"Total Episodes:   {len(returns)}")
print(f"Mean Return:      {np.mean(returns):.2f} ± {np.std(returns):.2f}")
print(f"Median Return:    {np.median(returns):.2f}")
print(f"Min / Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
print(f"25th Percentile:  {np.percentile(returns, 25):.2f}")
print(f"75th Percentile:  {np.percentile(returns, 75):.2f}")
print(f"95th Percentile:  {np.percentile(returns, 95):.2f}")

# 4. Plot the Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].hist(returns, bins=30, color='skyblue', edgecolor='black')
ax[0].set_title("Episode Return Distribution")
ax[0].set_xlabel("Cumulative Reward")
ax[0].set_ylabel("Frequency")

ax[1].scatter(episode_lengths, returns, alpha=0.5, color='orange')
ax[1].set_title("Return vs. Episode Length")
ax[1].set_xlabel("Episode Steps")
ax[1].set_ylabel("Cumulative Reward")

plt.tight_layout()
plt.show()