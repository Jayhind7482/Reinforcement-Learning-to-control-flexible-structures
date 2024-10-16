import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from elasticEnv import OptimizedElasticaEnv

# Instantiate the custom environment
env = OptimizedElasticaEnv()

# Wrap the environment for Stable-Baselines3
# The Monitor wrapper is used to track rewards and other useful information
env = Monitor(env)
env = DummyVecEnv([lambda: env])  # Stable-Baselines3 requires a vectorized environment



# Initialize the PPO model with the environment
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_elastica_tensorboard/",
            learning_rate=3e-4,  # Adjust learning rate
            n_steps=2048,        # Increase steps per update
            batch_size=64,       # Adjust batch size
            n_epochs=10,         # Increase number of epochs
            gamma=0.99,          # Discount factor
            gae_lambda=0.95,     # GAE lambda parameter
            clip_range=0.2,      # Clipping parameter
            ent_coef=0.01)       # Entropy coefficient

# Create a callback for saving checkpoints
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                         name_prefix='ppo_elastica')

# Train the model with the callback
model.learn(total_timesteps=10000, callback=checkpoint_callback)

# Save the trained model
model.save("ppo_elastica")

# Load the trained model
model = PPO.load("ppo_elastica")

# Test the trained model with rendering
env = env.envs[0]  # Extract the base environment to interact directly with it
env.enable_render = True
obs = env.reset()
done, truncated = False, False

episode_reward = 0
step = 0

while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
    
    print(f"Step {step}: Action = {action}, State = {obs}, Reward = {reward}")
    step += 1

print(f"Episode finished. Total reward: {episode_reward}")

# Close the environment
env.close()
