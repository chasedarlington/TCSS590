import gymnasium as gym
from gymnasium.utils.play import play


def play_lunar_lander():
    env = gym.make(
        "LunarLander-v2",
        render_mode="rgb_array"
    )

    keys_to_action = {
        "args": 1,        # fire left engine
        "w": 2,        # fire main engine
        "d": 3,        # fire right engine
        "s": 0,        # do nothing
        "": 0,         # no key pressed
    }

    play(
        env,
        keys_to_action=keys_to_action,
        noop=0,
        fps=30,
    )

    env.close()


if __name__ == "__main__":
    play_lunar_lander()