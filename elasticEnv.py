import numpy as np
from scipy.integrate import solve_bvp
import gymnasium as gym
from gymnasium.spaces import Box
import numba
import pygame

@numba.jit(nopython=True)
def elastica_f(s, y, h, v):
    return np.vstack((y[1], h * np.sin(y[0]) - v * np.cos(y[0])))

@numba.jit(nopython=True)
def elastica_bc(ya, yb):
    return np.array([ya[0], yb[1]])

def elastica_solve(h, v, l, s):
    y0 = np.zeros((2, s.size), dtype=np.float32)
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
def calculate_reward(x_tip, y_tip, x_target, y_target):
    d = np.sqrt((x_tip - x_target)**2 + (y_tip - y_target)**2)
    return np.exp(-d)

class OptimizedElasticaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = Box(low=-100, high=100, shape=(13,), dtype=np.float32)
        self.target_space = Box(low=np.array([4.5, -1]), high=np.array([5.57, 1]), dtype=np.float32)
        
        self.l = 6
        self.s = np.linspace(0, self.l, 500, dtype=np.float32)
        self.num_timestep = 0
        self.h = -0.4
        self.v = 0.15
        self.x_target = 0
        self.y_target = 0
        
        self.screen_width = 800
        self.screen_height = 600
        self.zoom_factor = 60
        self.enable_render = False
        self.screen = None
        self.np_random = None  # Add this line
        self.reset()  # This will initialize self.np_random

    def step(self, action):
        self.num_timestep += 1
        scaled_action = np.array([
            action[0] * 0.15 + -0.05,
            action[1] * 0.2
        ], dtype=np.float32)
        self.h += scaled_action[0]
        self.v += scaled_action[1]
        sol = elastica_solve(self.h, self.v, self.l, self.s)
        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = elastica_compute(sol, self.l, self.s)

        observation = self._get_observation()
        reward = calculate_reward(self.X[-1], self.Y[-1], self.x_target, self.y_target)
        terminated = self._check_done()
        truncated = self._check_truncated()
        
        if self.enable_render:
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.x_target, self.y_target = self.target_space.sample()
        self.h = -0.4
        self.v = 0.15 if self.y_target <= 0 else -0.15
        
        sol = elastica_solve(self.h, self.v, self.l, self.s)
        self.X, self.Y, self.theta_dash_0, self.theta_dash_l, self.theta_l, self.E = elastica_compute(sol, self.l, self.s)
        self.num_timestep = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            self.X[-1], self.Y[-1], self.X[200], self.Y[200], self.X[400], self.Y[400],
            self.theta_l, self.theta_dash_0, self.theta_dash_l, self.E,
            self.x_target, self.y_target,
            np.sqrt((self.X[-1] - self.x_target)**2 + (self.Y[-1] - self.y_target)**2)
        ], dtype=np.float32)

    def _check_done(self):
        return bool(np.sqrt((self.X[-1] - self.x_target)**2 + (self.Y[-1] - self.y_target)**2) < 0.002)

    def _check_truncated(self):
        return bool(self.num_timestep > 19)

    def render(self):
        if not self.enable_render:
            self.enable_render = True
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Optimized Elastica")
        self._render_frame()

    def _render_frame(self):
        self.screen.fill((255, 255, 255))
        offset_x = (self.screen_width - 10 * self.zoom_factor) / 2
        offset_y = (self.screen_height - 1.5 * self.zoom_factor) / 2
        
        points = [(x * self.zoom_factor + offset_x, y * self.zoom_factor + offset_y) for x, y in zip(self.X, self.Y)]
        pygame.draw.lines(self.screen, (0, 0, 0), False, points)
        
        tip_x, tip_y = self.X[-1] * self.zoom_factor + offset_x, self.Y[-1] * self.zoom_factor + offset_y
        pygame.draw.line(self.screen, (255, 0, 0), (tip_x, tip_y), (tip_x + 50 * np.sign(self.h), tip_y), 3)
        pygame.draw.line(self.screen, (0, 255, 0), (tip_x, tip_y), (tip_x, tip_y + 50 * np.sign(self.v)), 3)
        
        base_x, base_y = self.X[0] * self.zoom_factor + offset_x, self.Y[0] * self.zoom_factor + offset_y
        pygame.draw.line(self.screen, (0, 0, 0), (base_x, base_y), (base_x, base_y + 25), 3)
        pygame.draw.line(self.screen, (0, 0, 0), (base_x, base_y), (base_x, base_y - 25), 3)
        
        target_x, target_y = self.x_target * self.zoom_factor + offset_x, self.y_target * self.zoom_factor + offset_y
        pygame.draw.circle(self.screen, (255, 0, 0), (int(target_x), int(target_y)), 5)
        
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Timesteps: {self.num_timestep}", True, (0, 0, 0))
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 30, 120))
        
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
        self.screen = None
        self.enable_render = False