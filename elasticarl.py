import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from elasticEnv import OptimizedElasticaEnv

def make_env(rank, seed=0):
    def _init():
        env = OptimizedElasticaEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def cleanup_env(env):
    if isinstance(env, VecNormalize):
        env.close()
    elif hasattr(env, 'close'):
        env.close()

def train_model(total_timesteps=10000):
    n_envs = 4
    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn'))
    eval_env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(4)], start_method='spawn'))

    model = SAC('MlpPolicy', env, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        return model
    except KeyboardInterrupt:
        print("\nTraining interrupted. Calculating statistics...")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        try:
            num_timestep = model.env.get_attr('num_timestep')[0]
            if num_timestep > 0:
                total_episodes = model.num_timesteps // num_timestep
                episodes_per_second = total_episodes / total_time
            else:
                total_episodes = 0
                episodes_per_second = 0
        except (AttributeError, IndexError, ZeroDivisionError):
            total_episodes = 0
            episodes_per_second = 0

        print(f"\nTraining Statistics:")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Total timesteps: {model.num_timesteps}")
        print(f"Total episodes: {total_episodes}")
        print(f"Episodes per second: {episodes_per_second:.2f}")
        print(f"Frames per second: {model.num_timesteps / total_time:.2f}")

        cleanup_env(env)
        cleanup_env(eval_env)

def evaluate_model(model, env, episodes=50):
    env.enable_render = True
    env.render()
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                action, _ = model.predict(state, deterministic=True)
                print(f"Episode {episode}, Step {step}: Action = {action}")
                state, reward, done, truncated, info = env.step(action)
                print(f"  State = {state[:2]}, Reward = {reward}, Done = {done}, Truncated = {truncated}")
                score += reward
                env.render()
                step += 1
                if step >= 20:  # Add a step limit to prevent infinite loops
                    print("Step limit reached")
                    break
            print(f'Episode: {episode} Score: {score}')
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # Test environment
    print("Testing environment...")
    env = OptimizedElasticaEnv()
    env.reset()
    env.close()

    # Train model
    print("\nTraining model...")
    start_time = time.time()
    model = train_model()
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time:.2f} seconds")

    # Save model
    model_path = os.path.join('Training', 'Saved Models', 'OptimizedElastica_SAC')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate model
    print("\nEvaluating model...")
    eval_env = OptimizedElasticaEnv()
    evaluate_model(model, eval_env)

    eval_env.close()
