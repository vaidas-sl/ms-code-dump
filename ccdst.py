from typing import Optional
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from scipy.interpolate import interp1d


class DeepSeaTreasure(gym.Env, EzPickle):
   metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
   def __init__(self, render_mode: Optional[str] = None):
       EzPickle.__init__(self, render_mode)
       self.render_mode = render_mode
       self.reward_space = Box(
           low=np.array([-1, 0]),
           high=np.array([-1, 7]),
           dtype=np.float32,
       )
       self.reward_dim = 2
       self.observation_space = Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
       self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
       self.current_state = np.array([0.0, 0.0], dtype=np.float32)

       self.sea_map_x = 11
       self.sea_map_y = 11

       self.window_size = (min(64 * self.sea_map_x, 512), min(64 * self.sea_map_x, 512))

       self.window = None
       self.clock = None

       self.OVAL_CENTER = (self.window_size[0]*1 , 0)
       self.OVAL_WIDTH = self.window_size[0]*2
       self.OVAL_HEIGHT = self.window_size[0] * 1.6
       self.OVAL_RECT = pygame.Rect(0, 0, self.OVAL_WIDTH, self.OVAL_HEIGHT)

       self.points_x = [0, 0.35, 1.03, 2.00, 3.21, 4.60]
       self.points_y = [1, 1.97, 2.82, 3.46, 3.83, 3.86]
       #[1, 4, 5, 5.58, 5.9, 6]
       #-1,-2,-3,-4,   -5,  -6
       self.f_cubic = interp1d(self.points_x, self.points_y, kind='linear')
  
   def calculate_slope_and_intercept(self, start_point, end_point):
       x1, y1 = start_point
       x2, y2 = end_point

       # Handle vertical lines
       if x1 == x2:
           return None, y1  # Return None for slope and y1 as the intercept

       m = (y2 - y1) / (x2 - x1)
       b = y1 - m * x1

       return m, b

   def find_intersection_point(self, start_point, end_point, epsilon=0.01):
       m1, b1 = self.calculate_slope_and_intercept(start_point, end_point)

       for i in range(len(self.points_x) - 1):
           p1, p2 = (self.points_x[i], self.points_y[i]), (self.points_x[i+1], self.points_y[i+1])
           m2, b2 = self.calculate_slope_and_intercept(p1, p2)

           # Handle parallel lines
           if m1 == m2 or (m1 is None and m2 is None):
               continue

           # Find intersection
           if m1 is not None and m2 is not None:
               x_intersection = (b2 - b1) / (m1 - m2)
               y_intersection = m1 * x_intersection + b1
           elif m1 is None and m2 is not None:
               x_intersection, y_intersection = start_point[0], m2 * start_point[0] + b2
           elif m2 is None and m1 is not None:
               x_intersection, y_intersection = p1[0], m1 * p1[0] + b1
           else:
               continue  # Skip if both lines are vertical


           if (min(p1[0], p2[0]) - epsilon <= x_intersection <= max(p1[0], p2[0]) + epsilon and
               min(p1[1], p2[1]) - epsilon <= y_intersection <= max(p1[1], p2[1]) + epsilon and
               min(start_point[0], end_point[0]) - epsilon <= x_intersection <= max(start_point[0], end_point[0]) + epsilon and
               min(start_point[1], end_point[1]) - epsilon <= y_intersection <= max(start_point[1], end_point[1]) + epsilon):
               return round(x_intersection, 2), round(y_intersection, 2)
       return None

   def render(self):
       light_blue = (173,216,230)
       yellow = (255,255,0)
       brown = (165,42,42)
       red_square = pygame.Surface([4,4])
       red_square.fill((255, 0, 0))

       if self.window is None:
           pygame.init()
           if self.render_mode == "human":
               pygame.display.init()
               pygame.display.set_caption("CCDST")
               self.window = pygame.display.set_mode(self.window_size)
           else:
               self.window = pygame.Surface(self.window_size)

           if self.clock is None:
               self.clock = pygame.time.Clock()

       def scale_point(x, y, max_x, max_y):
           return x * self.window_size[0] / max_x, y * self.window_size[1] / max_y

       self.window.fill(light_blue)

       for x_pixel in range(self.window_size[0]):
           graph_x = x_pixel / (self.window_size[0]) * 10
           if graph_x <= 4.6: 
               graph_y = self.f_cubic(graph_x)
               pixel_y = scale_point(graph_x, graph_y, 10, 10)[1]
               pygame.draw.line(self.window, yellow, (x_pixel, pixel_y), (x_pixel, self.window_size[1]))
           else:
               pygame.draw.line(self.window, brown, (x_pixel, 3.86 * self.window_size[1] / 10), (x_pixel, self.window_size[1]))
          
      
       submarine_pos = scale_point(self.current_state[0], self.current_state[1], 10, 10)
       self.window.blit(red_square, submarine_pos)

       if self.render_mode == "human":
           pygame.event.pump()
           pygame.display.update()
           self.clock.tick(self.metadata["render_fps"])
       elif self.render_mode == "rgb_array":
           return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))


   def _get_state(self):
       state = self.current_state.copy()
       return state


   def reset(self, seed=None):
       super().reset(seed=seed)
       self.current_state = np.array([0.0, 0.0], dtype=np.float32)
       self.step_count = 0.0
       state = self._get_state()
       if self.render_mode == "human":
           self.render()
       return state, {}

   def step(self, action):
       action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
       angle = action[0]
       angle = np.pi*angle/3+2*np.pi/3

       # Calculate new position
       dx = np.sin(angle)
       dy = -np.cos(angle)  # Negative as moving down reduces y

       intersection_point = self.find_intersection_point(self.current_state, self.current_state + np.array([dx, dy]))
       self.current_state += np.array([dx, dy])
       self.step_count += 1
       treasure_collected = False

       # Initialize rewards
       r1 = -1
       r2 = 0

       if intersection_point:
           intersection_x, intersection_y = intersection_point
           distance = np.sqrt(intersection_x**2 + intersection_y**2)
           distance = max(distance, 1)
           r2 = round(np.sqrt(25 - (distance - 6)**2) + 1, 2)
           treasure_collected = True
       reward = np.array([r1, r2])

       # Check for episode termination
       done = treasure_collected or self.step_count >= 6
       return self.current_state, reward, done, False, {}

   def close(self):
       if self.window is not None:
           pygame.display.quit()
           pygame.quit()
