import multiprocessing
from multiprocessing import freeze_support
multiprocessing.set_start_method('spawn' , force = True)
import os
from env import Elastica_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import  SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import optuna
import pickle
from stable_baselines3.common.env_checker import check_env

env = Elastica_env()
# Function to create and seed each environment
def make_custom_env(rank: int, seed: int = 0):
    def _init():
        env = Elastica_env()  
        env.seed(seed + rank)
        return env
    return _init


if __name__ == '__main__':
    freeze_support()
    # Move all your main execution code here
    env = Elastica_env()
    env.enable_render = True
    log_path = os.path.join('Training', 'Logs')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    eval_env = Elastica_env()
    eval_env = Monitor(eval_env, log_path)

    # Set up the number of environments
    num_cpu = 12  # Number of environments to run in parallel (number of CPU cores)
    env2 = SubprocVecEnv([make_custom_env(i) for i in range(num_cpu)], start_method='spawn')
    eval_env1 = SubprocVecEnv([make_custom_env(i) for i in range(num_cpu)], start_method='spawn')

    results_path = os.path.join('Training', 'OptunaResults')
    model_save_path = os.path.join('Training', 'SavedModels', 'SAC')
    optuna_file = os.path.join(results_path, 'optuna_results.pkl')
    model_file = os.path.join(model_save_path, "final_sac_model.zip")

    if os.path.exists(optuna_file) and os.path.exists(model_file):
        print("Loading saved Optuna results and trained model...")
        # Load Optuna results
        with open(optuna_file, 'rb') as f:
            loaded_study = pickle.load(f)
        best_parameters = loaded_study.best_params
        print("Loaded best parameters:", best_parameters)

        # Load trained model
        model = SAC.load(model_file, env=env2)
        print("Model loaded successfully")

        # Continue training the loaded model
        eval_callback = EvalCallback(eval_env, eval_freq=200, deterministic=True)
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=model_save_path, name_prefix="sac_model")

        model.learn(total_timesteps=50000, callback=[eval_callback, checkpoint_callback])

        # Save the updated model
        model.save(model_file)
        print("Updated model saved successfully")

    else:
        print("No saved files found. Running optimization and training...")
        # Define the objective function
        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            ent_coef = 'auto'  # Use automatic entropy coefficient
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            tau = trial.suggest_float('tau', 0.005, 0.05)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            buffer_size = trial.suggest_categorical('buffer_size', [10000, 50000, 100000, 200000])
            
            model = SAC('MlpPolicy', env2, verbose=0, learning_rate=learning_rate,
                        ent_coef=ent_coef, gamma=gamma, tau=tau, batch_size=batch_size,
                        buffer_size=buffer_size, train_freq=1, gradient_steps=1,
                        learning_starts=1000)
            
            eval_callback = EvalCallback(eval_env1, eval_freq=100, n_eval_episodes=5, deterministic=True, verbose=0)
            model.learn(total_timesteps=1000, callback=eval_callback)

            mean_reward, _ = evaluate_policy(model, eval_env1, n_eval_episodes=10, deterministic=True)
            return mean_reward

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, n_jobs=1)
        best_parameters = study.best_params
        print("Best parameters:", best_parameters)

        # Save Optuna results
        os.makedirs(results_path, exist_ok=True)
        with open(optuna_file, 'wb') as f:
            pickle.dump(study, f)

        # Create and train the model with best parameters
        model = SAC('MlpPolicy', env2, verbose=0, learning_rate=best_parameters['learning_rate'],
                    ent_coef='auto', gamma=best_parameters['gamma'], tau=best_parameters['tau'], 
                    batch_size=best_parameters['batch_size'], buffer_size=best_parameters['buffer_size'],
                    train_freq=1, gradient_steps=1, learning_starts=1000)

        eval_callback = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=5, 
                                     deterministic=True, verbose=0)
        
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_save_path, 
                                                 name_prefix="sac_model", verbose=0)

        model.learn(total_timesteps= 200000, callback=[eval_callback, checkpoint_callback])

        # Save the final model
        model.save(model_file)
        print("Model saved successfully")

    # You can now use 'best_parameters' and 'loaded_model' (or 'model') for further operations
    # For example, you could run an evaluation here:
    # mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Evaluation code
    mean_reward, std_reward = evaluate_policy(model, eval_env1, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
