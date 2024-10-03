from gym.envs.registration import register

register(
    id="pong/Pong-v0",
    entry_point="pong.envs:PongEnv",
    max_episode_steps=400
)
