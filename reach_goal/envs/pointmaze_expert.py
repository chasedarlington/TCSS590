import numpy as np
import torch

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EXPLORATION_ACTIONS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

RIGHT_PATH = {(6, 3): (5, 3), (5, 3): (5, 4), (5, 4): (3, 4), (3, 4): (3, 2),
              (3, 2): (2, 2)}
LEFT_PATH = {(5, 3): (6, 3), (6, 3): (6, 1), (6, 1): (4, 1), (4, 1): (4, 2),
             (4, 2): (3, 2), (3, 2): (2, 2)}
EXPERT_PATH = {(5, 3): (6, 3), (6, 3): (6, 1), (6, 1): (4, 1), (4, 1): (4, 2),
             (4, 2): (3, 2), (3, 2): (2, 2), (5, 4): (3, 4), (3, 4): (3, 2)}

class WaypointController:
    """Agent controller to follow waypoints in the maze.

    Inspired by https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/waypoint_controller.py
    """

    def __init__(self, maze, gains={"p": 10.0, "d": -1.0}, waypoint_threshold=0.1):
        self.global_target_xy = np.empty(2)
        self.maze = maze

        # Hardcode goal and reset pos
        self.goal = np.array([-2, 2])

        self.gains = gains
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_targets = None

    def get_action(self, obs):
        if len(obs.shape) == 1:
            pass
        elif len(obs.shape) == 2:
            actions = []
            for i in range(obs.shape[0]):
                action = self.get_action(obs[i])
                actions.append(action)
            actions = np.array(actions)[None, ...]
            return actions
        else:
            raise ValueError("Invalid observation shape")
        # Check if we need to generate new waypoint path due to change in global target
        if (
                np.linalg.norm(self.global_target_xy - self.goal) > 1e-3
                or self.waypoint_targets is None
        ):
            # Convert xy to cell id
            achieved_goal_cell = tuple(
                self.maze.cell_xy_to_rowcol(obs[:2])
            )
            self.global_target_id = tuple(
                self.maze.cell_xy_to_rowcol(self.goal)
            )
            self.global_target_xy = self.goal

            self.waypoint_targets = EXPERT_PATH

            # Check if the waypoint dictionary is empty
            # If empty then the ball is already in the target cell location
            if self.waypoint_targets:
                self.current_control_target_id = self.waypoint_targets[
                    achieved_goal_cell
                ]
                self.current_control_target_xy = self.maze.cell_rowcol_to_xy(
                    np.array(self.current_control_target_id)
                )
            else:
                self.waypoint_targets[
                    self.current_control_target_id
                ] = self.current_control_target_id
                self.current_control_target_id = self.global_target_id
                self.current_control_target_xy = self.global_target_xy

        # Check if we need to go to the next waypoint
        dist = np.linalg.norm(self.current_control_target_xy - obs[:2])
        if (
                dist <= self.waypoint_threshold
                and self.current_control_target_id != self.global_target_id
        ):
            self.current_control_target_id = self.waypoint_targets[
                self.current_control_target_id
            ]
            # If target is global goal go directly to goal position
            if self.current_control_target_id == self.global_target_id:
                self.current_control_target_xy = self.global_target_xy
            else:
                self.current_control_target_xy = (
                        self.maze.cell_rowcol_to_xy(
                            np.array(self.current_control_target_id)
                        )
                        - np.random.uniform(size=(2,)) * 0.2
                )

        action = (
                self.gains["p"] * (self.current_control_target_xy - obs[:2])
                + self.gains["d"] * obs[2:]
        )
        action = np.clip(action, -1, 1)

        return action
    
if __name__ == "__main__":
    from pointmaze_env import PointMazeEnv
    env = PointMazeEnv(render_mode="human")
    controller = WaypointController(env.maze)
    obs, _ = env.reset()
    done = False
    while not done:
        obs_dict = env._get_obs(obs)
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        env.render()
    env.close()