import numpy as np
from scipy.integrate import solve_bvp
import gymnasium as gym
from gymnasium.spaces import Box
import numba
import pygame
import pickle
from scipy.spatial import cKDTree
from gymnasium.wrappers import NormalizeObservation

# Load the cheat sheet data
with open('cheat_sheet_data.pkl', 'rb') as f:
    cheat_sheet_data = pickle.load(f)

grid_points = cheat_sheet_data['grid_points']
theta_values = cheat_sheet_data['theta_values']
theta_prime_values = cheat_sheet_data['theta_prime_values']
kdtree = cheat_sheet_data['kdtree']

@numba.jit(nopython=True)
def elastica_f(s, y, h, v):
    return np.vstack((y[1], h * np.sin(y[0]) - v * np.cos(y[0])))

@numba.jit(nopython=True)
def elastica_bc(ya, yb):
    return np.array([ya[0], yb[1]])

def find_nearest_solution(h, v):
    _, index = kdtree.query([h, v])
    return theta_values[index], theta_prime_values[index]

def elastica_solve(h, v, l, s):
    nearest_theta, nearest_theta_prime = find_nearest_solution(h, v)
    y0 = np.vstack((nearest_theta, nearest_theta_prime))
    sol = solve_bvp(lambda s, y: elastica_f(s, y, h, v), elastica_bc, s, y0)
    return sol.sol(s).astype(np.float32)

@numba.jit(nopython=True)
def elastica_compute(sol, l, s):
    theta, dtheta_ds = sol[0], sol[1]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    dtheta_ds_squared = dtheta_ds**2
    
    x = np.cumsum(cos_theta) * (l / len(s))
    y = np.cumsum(sin_theta) * (l / len(s))
    
    e = 0.5 * np.sum(dtheta_ds_squared) * (l / len(s)) - np.sum(sin_theta) * (l / len(s)) + np.sum(1 - cos_theta) * (l / len(s))
    
    return x, y, dtheta_ds[0], dtheta_ds[-1], theta[-1], e

@numba.jit(nopython=True)
def calculate_reward(X_tip, Y_tip, x_target, y_target, E):
    distance = np.sqrt((X_tip - x_target)**2 + (Y_tip - y_target)**2)
    
    # Base reward for being close to the target
    reward = -distance
    
    return reward


class OptimizedElasticaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Update action space to be incremental
        self.action_space = Box(low=np.array([-3, -3]), high=np.array([3, 3]), dtype=np.float32)
        self.observation_space = Box(low=-100, high=100, shape=(13,), dtype=np.float32)
        # Update target space to match cheat sheet box
        self.target_space = Box(low=np.array([0.5, -0.4]), high=np.array([0.9, 0.4]), dtype=np.float32)
        
        self.l = 1  # length of the elastica
        self.s = np.linspace(0, self.l, 500, dtype=np.float32)
        self.num_timestep = 0
        # Initialize h and v to 0
        self.h = 0
        self.v = 0
        self.x_target = 0
        self.y_target = 0
        
        # Add these lines to define the limits for h and v
        self.h_limit = (-32, 32)
        self.v_limit = (-17, 17)

        self.screen_width = 800
        self.screen_height = 600
        self.zoom_factor = 60
        self.enable_render = False
        self.screen = None
        self.clock = None  # Add this line
        self.np_random = None  # Add this line
        self.reset()  # This will initialize self.np_random

        # Add these lines to store the cheat sheet data
        self.grid_points = grid_points
        self.theta_values = theta_values
        self.theta_prime_values = theta_prime_values
        self.kdtree = kdtree

        # Add these lines to improve reproducibility
        self.np_random = None
        self.seed()

        self.max_episode_steps = 19  # Add this line to define max steps

        # Remove the NormalizeObservation wrapper
        self = NormalizeObservation(self)

    def step(self, action):
        self.num_timestep += 1
            
        # Update h and v incrementally and clip them to their limits
        self.h = np.clip(self.h + action[0], self.h_limit[0], self.h_limit[1])
        self.v = np.clip(self.v + action[1], self.v_limit[0], self.v_limit[1])

        sol = elastica_solve(self.h, self.v, self.l, self.s)
        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = elastica_compute(sol, self.l, self.s)

        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_done()
        truncated = self._check_truncated()
        
        info = {}

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        # Initialize h and v to random values within their limits
        self.h = self.np_random.uniform(self.h_limit[0], self.h_limit[1])
        self.v = self.np_random.uniform(self.v_limit[0], self.v_limit[1])
        self.x_target, self.y_target = self.target_space.sample()
        
        sol = elastica_solve(self.h, self.v, self.l, self.s)
        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = elastica_compute(sol, self.l, self.s)
        self.num_timestep = 0
        
        info = {}

        return self._get_observation(), info

    def _get_observation(self):
        return np.array([
            self.X[-1], self.Y[-1], self.X[200], self.Y[200], self.X[400], self.Y[400],
            self.theta_l, self.theta_dash_0, self.theta_dash_l, self.E,
            self.x_target, self.y_target,
            np.sqrt((self.X[-1] - self.x_target)**2 + (self.Y[-1] - self.y_target)**2)
        ], dtype=np.float32)

    def _check_done(self):
        distance = np.sqrt((self.X[-1] - self.x_target)**2 + (self.Y[-1] - self.y_target)**2)
        return bool(distance < 0.01)  # Tightened success criterion

    def _check_truncated(self):
        return bool(self.num_timestep >= self.max_episode_steps)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Optimized Elastica")
            self.clock = pygame.time.Clock()
        return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            return None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        self.screen.fill((255, 255, 255))
        offset_x = (self.screen_width - 10 * self.zoom_factor) / 2
        offset_y = (self.screen_height - 1.5 * self.zoom_factor) / 2
        
        # Draw elastica
        points = [(x * self.zoom_factor + offset_x, self.screen_height - (y * self.zoom_factor + offset_y)) for x, y in zip(self.X, self.Y)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points, 2)
        
        # Draw tip forces
        tip_x, tip_y = self.X[-1] * self.zoom_factor + offset_x, self.screen_height - (self.Y[-1] * self.zoom_factor + offset_y)
        pygame.draw.line(self.screen, (255, 0, 0), (tip_x, tip_y), (tip_x + 50 * np.sign(self.h), tip_y), 3)
        pygame.draw.line(self.screen, (0, 255, 0), (tip_x, tip_y), (tip_x, tip_y - 50 * np.sign(self.v)), 3)
        
        # Draw base
        base_x, base_y = self.X[0] * self.zoom_factor + offset_x, self.screen_height - (self.Y[0] * self.zoom_factor + offset_y)
        pygame.draw.line(self.screen, (0, 0, 0), (base_x, base_y), (base_x, base_y + 25), 3)
        pygame.draw.line(self.screen, (0, 0, 0), (base_x, base_y), (base_x, base_y - 25), 3)
        
        # Draw target
        target_x, target_y = self.x_target * self.zoom_factor + offset_x, self.screen_height - (self.y_target * self.zoom_factor + offset_y)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(target_x), int(target_y)), 5)
        
        # Draw text information
        font = pygame.font.Font(None, 24)
        texts = [
            f"Timesteps: {self.num_timestep}",
            f"H: {self.h:.2f}",
            f"V: {self.v:.2f}",
            f"Tip X: {self.X[-1]:.2f}",
            f"Tip Y: {self.Y[-1]:.2f}",
            f"Target X: {self.x_target:.2f}",
            f"Target Y: {self.y_target:.2f}",
            f"Energy: {self.E:.2f}"
        ]
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(30)

        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def close(self):
        if self.screen:
            pygame.quit()
        self.screen = None
        self.clock = None  # Add this line

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _calculate_reward(self):
        return calculate_reward(self.X[-1], self.Y[-1], self.x_target, self.y_target, self.E)
