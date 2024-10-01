import os
import time
import numpy as np
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from elasticEnv import OptimizedElasticaEnv

def make_env(rank, seed=0):
    def _init():
        env = OptimizedElasticaEnv()
        env = Monitor(env)  # Add this line
        env.reset(seed=seed + rank)
        return env
    return _init


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)  # Changed from entropy_coef to ent_coef
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    tau = trial.suggest_float('tau', 0.005, 0.05, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 1000, 2000, 4000])
    target_update_interval = trial.suggest_categorical('target_update_interval', [1000, 5000, 10000])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4])
    buffer_size = trial.suggest_categorical('buffer_size', [100000, 200000, 500000, 1000000])

    # Change these lines
    n_envs = 8  # or more, depending on your CPU cores
    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn'))
    eval_env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(2)], start_method='spawn'))

    model = SAC('MlpPolicy', env, verbose=0, learning_rate=learning_rate,
                ent_coef=ent_coef, gamma=gamma, tau=tau, batch_size=batch_size,
                target_update_interval=target_update_interval, gradient_steps=gradient_steps,
                buffer_size=buffer_size)

    eval_callback = EvalCallback(eval_env, eval_freq=1000, deterministic=True, n_eval_episodes=5)
    model.learn(total_timesteps=10000, callback=eval_callback)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    return mean_reward

def train_model(best_params, total_timesteps=10000):
    n_envs = 8  # Use the same number as in the objective function
    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn'))
    eval_env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(2)], start_method='spawn'))

    # Create a copy of best_params and rename 'ent_coef' if it exists
    sac_params = best_params.copy()
    if 'entropy_coef' in sac_params:
        sac_params['ent_coef'] = sac_params.pop('entropy_coef')

    model = SAC('MlpPolicy', env, verbose=1, **sac_params)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, deterministic=True, n_eval_episodes=5)

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Calculating statistics...")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        total_episodes = model.num_timesteps // model.env.get_attr('num_timestep')[0]  # Get actual episode count
        episodes_per_second = total_episodes / total_time

        print(f"\nTraining Statistics:")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Total timesteps: {model.num_timesteps}")
        print(f"Total episodes: {total_episodes}")
        print(f"Episodes per second: {episodes_per_second:.2f}")
        print(f"Frames per second: {model.num_timesteps / total_time:.2f}")

    return model

def evaluate_model(model, env, episodes=50):
    env.render()  # Initialize the Pygame display
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            score += reward
        print(f'Episode: {episode} Score: {score}')

if __name__ == "__main__":
    # Test environment
    print("Testing environment...")
    env = OptimizedElasticaEnv()

    # Run a single Optuna trial
    print("\nRunning a single Optuna trial...")
    study = optuna.create_study(direction='maximize')
    start_time = time.time()
    study.optimize(objective, n_trials=1, n_jobs=1)
    end_time = time.time()
    best_params = study.best_params
    print("Best parameters:", best_params)
    print(f"Time taken for Optuna trial: {end_time - start_time:.2f} seconds")

    # Train model with best parameters
    print("\nTraining model with best parameters...")
    start_time = time.time()
    model = train_model(best_params)
    end_time = time.time()
    print(f"Time taken for training: {end_time - start_time:.2f} seconds")

    # Save model
    model_path = os.path.join('Training', 'Saved Models', 'OptimizedElastica_SAC')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate model
    print("\nEvaluating model...")
    env.enable_render = True
    evaluate_model(model, env)

    env.close()