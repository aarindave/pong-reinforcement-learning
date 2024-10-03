import math
import numpy as np
import pygame
import pygame.colordict
from pygame.locals import *
import random

import gym
from gym import spaces

# Initializes Pygame modules.
pygame.init()
pygame.font.init()

# Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400

MIN_HEIGHT = WINDOW_HEIGHT * 0.2


# Functions


def display_text(surface, text, x, y, color, size):
    font_object = pygame.font.Font(
        "/System/Library/Fonts/Avenir Next.ttc", size)
    text_object = font_object.render(text, False, color)
    surface.blit(text_object, (x, y))


# Classes


class Paddle:
    def __init__(self, x: float, color: tuple[int, int, int]):
        self.color: tuple[int, int, int] = color
        self.speed: float = 3
        self.score: int = 0
        self.size: list[int, int] = [20, 80]
        self.rect: pygame.Rect = Rect([x, WINDOW_HEIGHT // 2], self.size)
        self.rect.center = [x, WINDOW_HEIGHT // 2]

    def draw(self, surface: pygame.Surface):
        # Draw the paddle on the assigned surface.
        pygame.draw.rect(surface, self.color, self.rect, border_radius=5)
        # Draw the score of the game above the paddle.
        display_text(surface, str(self.score),
                     self.rect.x,
                     surface.get_height() * 0.05, self.color, 40)

    def move(self, speed_vector: tuple[float, float]):
        self.rect = self.rect.move(0, speed_vector[1] * -self.speed)

    def check_boundary(self):
        if self.rect.top < MIN_HEIGHT:
            self.rect.top = MIN_HEIGHT
        elif self.rect.bottom > WINDOW_HEIGHT:
            self.rect.bottom = WINDOW_HEIGHT

    def get_rect(self):
        return self.rect


class Ball:
    def __init__(self):
        self.RADIUS = 10
        self.DIAMETER = self.RADIUS * 2

        self.initial_vector: pygame.Vector2 = pygame.Vector2.normalize(
            pygame.Vector2(
                random.uniform(0.3, 0.7) * random.choice((-1, 1)),
                random.uniform(0.3, 0.7) * random.choice((-1, 1))
            )
        )
        self.velocity: pygame.Vector2 = pygame.Vector2(5, 5)

        self.rect: pygame.Rect = Rect(
            WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, self.DIAMETER, self.DIAMETER)
        self.rect.center = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]

    def draw(self, surface):
        pygame.draw.circle(surface, pygame.colordict.THECOLORS["white"],
                           self.rect.center, self.RADIUS)

    def move(self):
        self.rect = self.rect.move(
            self.initial_vector.x * self.velocity.x,
            self.initial_vector.y * self.velocity.y
        )

    def collide(self, window_size, left_paddle: Paddle, right_paddle: Paddle):
        """
        The function returns `True` if the round ends.
        """

        # If the ball collides with either the left or right paddle,
        # then the ball will invert its direction on the x-axis.
        if self.rect.collidelist([left_paddle.rect, right_paddle.rect]) != -1:
            self.velocity.x = -self.velocity.x
            if self.rect.colliderect(left_paddle.rect):
                self.rect.left = left_paddle.rect.right + 1
            else:
                self.rect.right = right_paddle.rect.left - 1

        # If the ball's y-coordinate is out of bounds, then the ball
        # will adjust its y-coordinate to be within the bounds, and
        # the function will return `False` to prevent the episode from
        # terminating.
        if self.rect.top < MIN_HEIGHT:
            self.rect.top = MIN_HEIGHT + 1
            self.velocity.y = -self.velocity.y
        if self.rect.bottom > window_size[1]:
            self.rect.bottom = window_size[1] - 1
            self.velocity.y = -self.velocity.y

        # If the ball's x-coordinate is out of bounds, then one point
        # will be awarded to the paddle on the opposite side, and the
        # function will return `True` to terminate the episode.
        if self.rect.left < 0:
            right_paddle.score += 1
            self.rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
            return True
        if self.rect.right > window_size[0]:
            left_paddle.score += 1
            self.rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
            return True

        return False


class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=5):
        # Define some constants for colors.
        self.BLACK = (20, 25, 40)
        self.BLUE = (0, 127, 255)
        self.RED = (255, 92, 89)
        # Derive the current window's width and window height.
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.size = size

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_width - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.window_width - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "up", "down", and "stay"
        self.action_space = spaces.Discrete(3)

        self._action_to_direction = {
            0: np.array([0, -1]),  # Left paddle moves up.
            1: np.array([0, 1]),  # Left paddle moves down.
            2: np.array([0, 0]),  # Left paddle stays.
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Choose the agent's location at the center of the window.
        self._agent_location = np.array(
            [self.window_width // 2, self.window_height // 2])
        # Choose the target's location to be the agent's location.
        self._target_location = np.array(
            [self.window_width // 2, self.window_height // 2])

        # Creates the Paddle objects.
        self.blue_paddle = Paddle(self.window_width * 0.1, self.BLUE)
        self.red_paddle = Paddle(self.window_width * 0.9, self.RED)
        self.ball = Ball()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # The blue paddle moves up and down at a different speed depending on how
        # far the ball's y-coordinate is from the blue paddle's y-coordinate.
        distance_y = self.blue_paddle.rect.centery - self.ball.rect.centery
        change_y = (2 / (1 + math.e ** (-0.1 * distance_y))) - 1
        blue_paddle_direction = [0, change_y]

        # Map the action (element of {0, 1, 2}) to the direction the paddles move.
        red_paddle_direction = self._action_to_direction[action]

        # Moves all of the objects in the game.
        self.blue_paddle.move(blue_paddle_direction)
        self.red_paddle.move(red_paddle_direction)
        self.ball.move()

        # Detects if the ball hit the edge.
        self.blue_paddle.check_boundary()
        self.red_paddle.check_boundary()

        terminated = self.ball.collide(
            (self.window_width, self.window_height),
            self.blue_paddle, self. red_paddle
        )

        reward = 0
        if terminated:
            reward = 2 if self.ball.rect.centerx < self.window_width // 2 else -1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            # Initializes Pygame modules.
            pygame.display.init()
            # Creates a Pygame window.
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height))
            pygame.display.set_caption("Reinforcement Learning - Pong")
        if self.clock is None and self.render_mode == "human":
            # Initializes a Pygame clock.
            self.clock = pygame.time.Clock()

        # Create a blank Surface object to draw objects on.
        canvas = pygame.Surface((self.window_width, self.window_height))
        # Make the window's background black.
        canvas.fill(self.BLACK)

        # The paddles are drawn.
        self.blue_paddle.draw(canvas)
        self.red_paddle.draw(canvas)
        # The "agent" is drawn.
        self.ball.draw(canvas)
        # Draws a minimum height line
        self.draw_dashed_line(canvas, 25)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return self.close()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_dashed_line(self, surface: pygame.Surface, dash_length: int):
        """
        A dashed line is drawn on top of the Pygame window.
        """
        for dash in range(0, surface.get_width(), dash_length * 2):
            pygame.draw.line(surface, "white", (dash, MIN_HEIGHT),
                             (dash + dash_length, MIN_HEIGHT), 5)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
