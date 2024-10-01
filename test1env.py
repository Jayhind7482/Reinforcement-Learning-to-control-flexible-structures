import pickle
from elasticEnv import OptimizedElasticaEnv

env = OptimizedElasticaEnv()
try:
    pickle.dumps(env)
    print("Environment is serializable")
except Exception as e:
    print("Environment is not serializable:", e)

import multiprocessing

num_cpu_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores available: {num_cpu_cores}")

from stable_baselines3.common.env_checker import check_env

env = OptimizedElasticaEnv()
check_env(env)
env.close()