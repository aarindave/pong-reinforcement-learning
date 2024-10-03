import gym
import pong
from gym.wrappers import FlattenObservation
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the environment with render mode specified
env = gym.make("pong/Pong-v0", render_mode="human")

# Initialize the environment to get the initial state
env.action_space.seed(123)
observation, info = env.reset(seed=123)

# Run a few steps in the environment with random actions
for _ in range(99_999):
    # Render the environment for visualization.
    env.render()
    # Take a random action.
    action = env.action_space.sample()

    # Take a step in the environment
    step_result = env.step(action)

    # Check the number of values returned and unpack accordingly
    if len(step_result) == 4:
        next_state, reward, done, info = step_result
        terminated = False
    else:
        next_state, reward, done, truncated, info = step_result
        terminated = done or truncated

    if terminated:
        # Reset the environment if the episode is finished.
        state = env.reset()

# Close the environment when done.
env.close()
