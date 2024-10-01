import time
import numpy as np
from elasticEnv import OptimizedElasticaEnv

def run_env_test(num_episodes=100, steps_per_episode=100):
    env = OptimizedElasticaEnv()
    
    total_time = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_start = time.time()
        
        for step in range(steps_per_episode):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        episode_time = time.time() - episode_start
        total_time += episode_time
        
        if episode % 10 == 0:
            print(f"Episode {episode} completed in {episode_time:.4f} seconds")
    
    avg_time = total_time / num_episodes
    print(f"\nAverage time per episode: {avg_time:.4f} seconds")
    print(f"Total time for {num_episodes} episodes: {total_time:.4f} seconds")

if __name__ == "__main__":
    run_env_test()