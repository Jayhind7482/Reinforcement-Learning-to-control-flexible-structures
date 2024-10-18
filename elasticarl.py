import os
import time
import optuna
import cv2
import numpy as np
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

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    tau = trial.suggest_float('tau', 0.005, 0.05, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_categorical('buffer_size', [100000, 200000, 500000, 1000000])
    learning_starts = trial.suggest_categorical('learning_starts', [1000, 5000, 10000, 20000])
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4, 8])

    n_envs = 4
    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn'))
    eval_env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(4)], start_method='spawn'))

    model = SAC('MlpPolicy', env, verbose=0, learning_rate=learning_rate,
                ent_coef=ent_coef, gamma=gamma, tau=tau, batch_size=batch_size,
                buffer_size=buffer_size, learning_starts=learning_starts,
                train_freq=train_freq, gradient_steps=gradient_steps)

    eval_callback = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=10, deterministic=True)
    model.learn(total_timesteps=1000, callback=eval_callback)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    try:
        return mean_reward
    finally:
        cleanup_env(env)
        cleanup_env(eval_env)

def train_model(best_params, total_timesteps=100000):
    n_envs = 4
    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='spawn'))
    eval_env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(4)], start_method='spawn'))

    model = SAC('MlpPolicy', env, verbose=1, **best_params)
    eval_callback = EvalCallback(eval_env, eval_freq=50000, n_eval_episodes=10, deterministic=True)

    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        return model
    except KeyboardInterrupt:
        print("\n\033[93mTraining interrupted. Calculating statistics...\033[0m")
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

        print("\n\033[1m\033[94mTraining Statistics:\033[0m")
        print(f"\033[1mTotal training time:\033[0m {total_time:.2f} seconds")
        print(f"\033[1mTotal timesteps:\033[0m {model.num_timesteps}")
        print(f"\033[1mTotal episodes:\033[0m {total_episodes}")
        print(f"\033[1mEpisodes per second:\033[0m {episodes_per_second:.2f}")
        print(f"\033[1mFrames per second:\033[0m {model.num_timesteps / total_time:.2f}")

        cleanup_env(env)
        cleanup_env(eval_env)

def evaluate_model(model, env, episodes=50, save_video=True):
    env.enable_render = True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join('Training', 'Videos', 'evaluation.mp4')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (env.screen_width, env.screen_height))
    
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            done = False
            truncated = False
            score = 0
            step_count = 0
            while not (done or truncated):
                action, _ = model.predict(state, deterministic=True)
                state, reward, done, truncated, _ = env.step(action)
                score += reward
                step_count += 1
                frame = env.render()
                if save_video:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_rgb)
                if step_count >= env.max_episode_steps:
                    break
            print(f'\033[1mEpisode:\033[0m {episode:3d} | \033[1mScore:\033[0m {score:8.2f} | \033[1mSteps:\033[0m {step_count:4d}')
    except KeyboardInterrupt:
        print("\n\033[93mEvaluation interrupted by user.\033[0m")
    finally:
        env.close()
        if save_video:
            out.release()
            print(f"\n\033[92mEvaluation video saved to:\033[0m {video_path}")

if __name__ == "__main__":
    print("\033[1m\033[95mTesting environment...\033[0m")
    # Test environment
    env = OptimizedElasticaEnv()
    env.reset()
    env.close()

    print("\n\033[1m\033[95mRunning Optuna study...\033[0m")
    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3, n_jobs=1)  # Adjust n_trials and n_jobs as needed
    best_params = study.best_params
    print("\033[1mBest parameters:\033[0m", best_params)

    print("\n\033[1m\033[95mTraining model with best parameters...\033[0m")
    # Train model with best parameters
    start_time = time.time()
    model = train_model(best_params)
    end_time = time.time()
    print(f"\033[1mTime taken for training:\033[0m {end_time - start_time:.2f} seconds")

    # Save model
    model_path = os.path.join('Training', 'Saved Models', 'OptimizedElastica_SAC')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\033[92mModel saved to:\033[0m {model_path}")

    print("\n\033[1m\033[95mEvaluating model...\033[0m")
    # Evaluate model
    eval_env = OptimizedElasticaEnv()
    eval_env.enable_render = True  # Ensure rendering is enabled
    evaluate_model(model, eval_env, save_video=True)

    eval_env.close()
