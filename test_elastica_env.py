from env import Elastica_env
from stable_baselines3 import SAC
import cv2
import numpy as np
import pygame
import os

env = Elastica_env()

# Update the model loading path
model_save_path = os.path.join('Training', 'SavedModels', 'SAC')
model_file = os.path.join(model_save_path, "final_sac_model.zip")

# Load the model
model = SAC.load(model_file, env=env)
print(f"Model loaded from: {model_file}")

env.enable_render = True

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('elastica_simulation.mp4', fourcc, 30.0, (int(env.screen_width), int(env.screen_height)))

episodes = 50
for episode in range(1, episodes+1):
    state, info = env.reset()
    done = False
    score = 0 
    truncation = False
    test = []
    n_score = []
    dis = []
    
    # Render initial state
    env.render(True)
    pygame.display.flip()
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = cv2.transpose(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)
    
    while not (done or truncation):
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, truncation, info = env.step(action)
        n_score.append(reward)
        dis.append(state[-1])
        score += reward
        test.append(done)
        
        # Render and capture frame for every timestep
        env.render(True)
        pygame.display.flip()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = cv2.transpose(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    print(f'Episode: {episode}, Score: {score:.4f}, Steps: {len(test)}')

# Release the video writer
video.release()
env.close()
