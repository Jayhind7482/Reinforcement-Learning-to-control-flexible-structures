import numpy as np
from scipy.integrate import solve_bvp
import gymnasium as gym
from gymnasium.spaces import Box
import pygame
import pickle
import math
from scipy.spatial import cKDTree
import pygame.gfxdraw  # For smoother graphics

class Elastica_env(gym.Env):
    def __init__(self):
        super(Elastica_env, self).__init__()
        self.action_space = Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([1, 1], dtype=np.float32))
        self.observation_space = Box(low=np.float32(-100), high=np.float32(100), shape=(15,), dtype=np.float64)
        
        # Update target space to match cheat sheet
        self.target = Box(low=np.array([0.5, -0.4], dtype=np.float32), high=np.array([0.9, 0.4], dtype=np.float64))
        self.reward = 0
        self.x = []
        self.y = []
        self.screen_width = 800.0
        self.screen_height = 600.0
        self.zoom_factor = 60.0
        self.enable_render = False
        self.h = 0.0
        self.v = 0.0
        self.num_timestep = 0

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
        return [seed]

    def step(self, action):
        self.num_timestep += 1
        # Map actions from [-1, 1] to their respective ranges
        self.h = action[0] * 16 - 16  # Map [-1, 1] to [-32, 0]
        self.v = action[1] * 17  # Map [-1, 1] to [-17, 17]
        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = self.elastica(self.h, self.v)

        new_observation = self.get_observation()
        self.render(self.enable_render)
        self.reward = self.score()
        done = self.get_done()
        truncated = self.num_timestep >= 10  # Add a maximum episode length
        info = {}
        return new_observation, self.reward, done, truncated, info

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

        # Update x and y calculations
        self.x = np.cumsum(y1) * (l / 500)
        self.y = np.cumsum(y2) * (l / 500)

        e = 0.5 * np.sum(y3) * (l / 500) - v * np.sum(y2) * (l / 500) + h * np.sum(1 - y1) * (l / 500)

        return self.x, self.y, dtheta_ds[0], dtheta_ds[-1], theta[-1], e

    def get_observation(self):
        self.x_tip = self.X[-1]
        self.y_tip = self.Y[-1]
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        return np.array([
            self.x_tip, self.y_tip,
            self.X[100], self.Y[100], self.X[300], self.Y[300],
            self.theta_l, self.theta_dash_0, self.theta_dash_l, self.E,
            self.x_target, self.y_target, d,
            self.h, self.v  # Include current action values
        ], dtype=np.float64)

    def score(self):
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        reward = -d  # Negative distance as base reward
        
        # Add a small reward for getting closer to the target
        if d < self.previous_distance:
            reward += 0.1
        

        # Add a larger reward for reaching the target
        if d < 0.04:
            reward += 0.5

        # Add a larger reward for reaching the target
        if d < 0.02:
            reward += 1
        
        self.previous_distance = d
        return reward

    def get_done(self):
        d = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        return d < 0.002  # Done if very close to target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.num_timestep = 0
        targ = self.target.sample()
        self.x_target = targ[0]
        self.y_target = targ[1]
        
        # Initialize h and v to zero
        self.h = 0
        self.v = 0

        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = self.elastica(self.h, self.v)
        observation = self.get_observation()
        self.previous_distance = math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target)
        info = {}
        return observation, info

    def render(self, enable_render):
        if not enable_render:
            return
        pygame.init()
        screen = pygame.display.set_mode((int(self.screen_width), int(self.screen_height)))
        pygame.display.set_caption("Elastica Simulation")
        screen.fill((255, 255, 255))
        
        # Increase zoom factor
        self.zoom_factor = 200.0
        
        # Adjust offsets to center the plot
        offset_x = self.screen_width / 2
        offset_y = self.screen_height / 2
        
        # Draw elastica
        points = [(self.X[i], self.Y[i]) for i in range(len(self.X))]
        pygame.draw.lines(screen, (0, 0, 0), False, [(point[0] * self.zoom_factor + offset_x, -point[1] * self.zoom_factor + offset_y) for point in points], 2)
        
        # Draw force vectors
        tip_x, tip_y = self.X[-1] * self.zoom_factor + offset_x, -self.Y[-1] * self.zoom_factor + offset_y
        pygame.draw.line(screen, (255, 0, 0), (tip_x, tip_y), (tip_x + 50 * np.sign(self.h), tip_y), 3)
        pygame.draw.line(screen, (0, 255, 0), (tip_x, tip_y), (tip_x, tip_y - 50 * np.sign(self.v)), 3)
        
        # Draw base
        base_x, base_y = self.X[0] * self.zoom_factor + offset_x, -self.Y[0] * self.zoom_factor + offset_y
        pygame.draw.line(screen, (0, 0, 0), (base_x, base_y), (base_x, base_y + 25), 3)
        pygame.draw.line(screen, (0, 0, 0), (base_x, base_y), (base_x, base_y - 25), 3)
        
        # Draw target box
        pygame.draw.rect(screen, (0, 0, 255), (
            0.5 * self.zoom_factor + offset_x,
            -0.4 * self.zoom_factor + offset_y,
            0.4 * self.zoom_factor,
            0.8 * self.zoom_factor
        ), 2)
        
        # Draw target point
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x_target * self.zoom_factor + offset_x), int(-self.y_target * self.zoom_factor + offset_y)), 5)
        
        # Add text information
        font = pygame.font.Font(None, 24)
        info_texts = [
            f"Timestep: {self.num_timestep}",
            f"Reward: {self.reward:.4f}",
            f"h: {self.h:.4f}",
            f"v: {self.v:.4f}",
            f"Tip position: ({self.x_tip:.4f}, {self.y_tip:.4f})",
            f"Target: ({self.x_target:.4f}, {self.y_target:.4f})",
            f"Distance: {math.hypot(self.x_tip - self.x_target, self.y_tip - self.y_target):.4f}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()

    def close(self):
        pygame.quit()
