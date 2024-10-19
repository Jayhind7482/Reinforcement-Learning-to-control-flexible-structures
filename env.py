import numpy as np
from scipy.integrate import solve_bvp
import gymnasium as gym
from gymnasium.spaces import Box
import pygame
import pickle
import math
from scipy.spatial import cKDTree

class Elastica_env(gym.Env):
    def __init__(self):
        super(Elastica_env, self).__init__()
        self.action_space = Box(low=np.array([-3, -3], dtype=np.float32), high=np.array([0.1, 3], dtype=np.float64))
        self.observation_space = Box(low=np.float32(-100), high=np.float32(100), shape=(13,), dtype=np.float64)
        
        # Update target space to match cheat sheet
        self.target = Box(low=np.array([0.5, -0.4], dtype=np.float32), high=np.array([0.9, 0.4], dtype=np.float64))
        self.num_timestep = 0
        self.reward = 0
        self.x = []
        self.y = []
        self.screen_width = 800.0
        self.screen_height = 600.0
        self.zoom_factor = 60.0
        self.enable_render = False
        self.h = -20  # Initialize h to the middle of the range
        self.v = -10  # Initialize v to the middle of the range

        # Pre-allocate arrays for elastica calculation
        self.s = np.linspace(0, 1, 500)
        self.y0 = np.zeros((2, self.s.size))

        # Load the cheat sheet data
        with open('cheat_sheet_data.pkl', 'rb') as f:
            self.cheat_sheet_data = pickle.load(f)
        
        # Create KD-tree for efficient nearest neighbor search
        self.kdtree = cKDTree(self.cheat_sheet_data['grid_points'])

    # Add the seed() method
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed
    def step(self, action):
        self.num_timestep += 1
        self.h += action[0]
        self.v += action[1]
        self.X, self.Y, self.theta_dash_0  ,self.theta_dash_l, self.theta_l , self.E = self.elastica(self.h, self.v)

        new_observation = self.get_observation()
        self.render(self.enable_render)
        self.reward = self.score()
        done = self.get_done()
        truncation = self.get_truncation()
        info = {}
        return new_observation, self.reward, done, truncation, info

    def elastica(self, h, v):
        l = 1
        s = self.s

        # Find the nearest neighbor in the cheat sheet
        _, nearest_index = self.kdtree.query([h, v])
        
        # Get the initial guess from the cheat sheet
        theta_guess = self.cheat_sheet_data['theta_values'][nearest_index]
        theta_prime_guess = self.cheat_sheet_data['theta_prime_values'][nearest_index]
        
        # Create the initial guess array
        y0 = np.array([theta_guess, theta_prime_guess])

        def f(s, y):
            return np.vstack((y[1], h * np.sin(y[0]) - v * np.cos(y[0])))

        def bc(ya, yb):
            return np.array([ya[0], yb[1]])

        sol = solve_bvp(f, bc, s, y0)
        theta = sol.sol(s)[0]
        dtheta_ds = sol.sol(s)[1]
        
        # Vectorize calculations
        y1 = np.cos(theta)
        y2 = np.sin(theta)
        y3 = (dtheta_ds)**2

        x = np.cumsum(y1) * (l / 500)
        y = np.cumsum(y2) * (l / 500)

        e = 0.5 * np.sum(y3) * (l / 500) - v * np.sum(y2) * (l / 500) + h * np.sum(1 - y1) * (l / 500)

        return x, y, dtheta_ds[0], dtheta_ds[-1], theta[-1], e

    def get_observation(self):
        self.x_tip = self.X[-1]
        self.y_tip = self.Y[-1]
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        return np.array([self.x_tip, self.y_tip, self.X[200], self.Y[200], self.X[400], self.Y[400], 
                         self.theta_l, self.theta_dash_0, self.theta_dash_l, self.E,
                         self.x_target, self.y_target, d], dtype=np.float64)

    def score(self):
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        return math.exp(-d)

    def get_done(self):
        done = False
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        if d < 0.002:
            done = True
        return done

    def get_truncation(self):
        truncation = False
        if self.num_timestep > 19 :
            truncation = True
        return truncation

    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        targ = self.target.sample()
        self.x_target = targ[0]
        self.y_target = targ[1]
        
        # Initialize h and v to the middle of their ranges
        self.h = -20
        self.v = 10

        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = self.elastica(self.h, self.v)
        self.num_timestep = 0
        self.reward = 0
        observation = self.get_observation()
        info = {}
        return observation, info

    def render(self, enable_render):
        if not enable_render:
            return
        pygame.init()
        screen = pygame.display.set_mode((int(self.screen_width), int(self.screen_height)))
        pygame.display.set_caption("Elastica")
        screen.fill((255, 255, 255))
        offset_x = (self.screen_width - 10 * self.zoom_factor) / 2
        offset_y = (self.screen_height - 1.5 * self.zoom_factor) / 2
        points = [(self.X[i], self.Y[i]) for i in range(len(self.X))]
        pygame.draw.lines(screen, (0, 0, 0), False, [(point[0] * self.zoom_factor + offset_x, point[1] * self.zoom_factor + offset_y) for point in points])
        pygame.draw.line(screen, (255, 0, 0), ((self.X[-1]) * self.zoom_factor + offset_x, (self.Y[-1]) * self.zoom_factor + offset_y), ((self.X[-1]) * self.zoom_factor + offset_x + 50 * np.sign(self.h), (self.Y[-1]) * self.zoom_factor + offset_y), 3)
        pygame.draw.line(screen, (0, 255, 0), ((self.X[-1]) * self.zoom_factor + offset_x, (self.Y[-1]) * self.zoom_factor + offset_y), ((self.X[-1]) * self.zoom_factor + offset_x, (self.Y[-1]) * self.zoom_factor + offset_y + 50 * np.sign(self.v)), 3)
        pygame.draw.line(screen, (0, 0, 0), ((self.X[0]) * self.zoom_factor + offset_x, (self.Y[0]) * self.zoom_factor + offset_y), ((self.X[0]) * self.zoom_factor + offset_x, (self.Y[0]) * self.zoom_factor + offset_y + 25), 3)
        pygame.draw.line(screen, (0, 0, 0), ((self.X[0]) * self.zoom_factor + offset_x, (self.Y[0]) * self.zoom_factor + offset_y), ((self.X[0]) * self.zoom_factor + offset_x, (self.Y[0]) * self.zoom_factor + offset_y - 25), 3)
        
        # Draw target box
        pygame.draw.rect(screen, (0, 0, 255), (
            0.5 * self.zoom_factor + offset_x,
            -0.4 * self.zoom_factor + offset_y,
            0.4 * self.zoom_factor,
            0.8 * self.zoom_factor
        ), 2)
        
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x_target * self.zoom_factor + offset_x), int(self.y_target * self.zoom_factor + offset_y)), 5)
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Timesteps: {self.num_timestep}", True, (0, 0, 0))
        screen.blit(score_text, (int(self.screen_width - score_text.get_width() - 30), 120))
        pygame.display.flip()

    def close(self):
        pygame.quit()
