import numpy as np
from elasticEnv import OptimizedElasticaEnv

def test_elastica_env():
    env = OptimizedElasticaEnv()
    env.enable_render = True  # Set to False if you don't want to render

    num_episodes = 100
    max_steps = 20

    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0

        print(f"Episode {episode + 1}")
        print(f"Initial observation: {observation}")
        print(f"Target: ({env.x_target:.2f}, {env.y_target:.2f})")

        for step in range(max_steps):
            action = env.action_space.sample()  # Random action
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            print(f"Step {step + 1}: Action: {action}, Reward: {reward:.4f}")

            if env.enable_render:
                env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} finished after {step + 1} steps")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final tip position: ({env.X[-1]:.2f}, {env.Y[-1]:.2f})")
        print("--------------------")

    env.close()

if __name__ == "__main__":
    test_elastica_env()
