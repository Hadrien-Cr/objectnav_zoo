# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys, os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

ZOO_ROOT = os.environ["ZOO_ROOT"]

sys.path.insert(
    0, ZOO_ROOT
)

import cv2
import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel

import objectnav_zoo.utils.pose as pu

from objectnav_zoo.agent.imagenav_agent.visualizer import NavVisualizer
from objectnav_zoo.core.abstract_agent import Agent
from objectnav_zoo.core.interfaces import DiscreteNavigationAction, Observations
from objectnav_zoo.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from objectnav_zoo.navigation_planner.discrete_planner import DiscretePlanner
from objectnav_zoo.mapping.semantic.instance_tracking_modules import InstanceMemory

from objectnav_zoo.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from objectnav_zoo.navigation_policy.language_navigation.languagenav_fbe import (
    LanguageNavFrontierExplorationPolicy,
)

from goat_matching import GoatMatching

# For visualizing exploration issues
debug_frontier_map = True



class GoatAgentModule(nn.Module):
    def __init__(
        self,
        config,
        matching: GoatMatching,
        instance_memory: Optional[InstanceMemory] = None,
    ):
        super().__init__()
        self.matching = matching
        self.instance_memory = instance_memory
        self.goal_inst = None
        self.instance_goal_found = False
        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[
                1
            ],            
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=config.AGENT.SEMANTIC_MAP.max_depth,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            exploration_type=config.AGENT.SEMANTIC_MAP.exploration_type,
        )
        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy
        )
        self.goal_policy_config = config.AGENT.SUPERGLUE
        self.instance_goal_found = False
        self.goal_inst = None

    def reset_sub_episode(self):
        self.instance_goal_found = False
        self.goal_inst = None

    def reset(self):
        self.reset_sub_episode()

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        reject_visited_targets=False,
        blacklist_target=False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        seq_obstacle_locations=None,
        seq_free_locations=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation, instance_segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)
            seq_object_goal_category: sequence of object goal categories of shape
             (batch_size, sequence_length, 1)
        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """

        # Reset the last channel of the local map each step when found_goal=False
        if matches is not None or confidence is not None:
            # TODO:
            init_local_map[:, 21][seq_found_goal[:, 0] == 0] *= 0.0

        stale_local_id_to_global_id_map = self.instance_memory.local_id_to_global_id_map.copy()
        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        
        assert seq_obs is not None
        assert seq_pose_delta is not None
        assert seq_dones is not None
        assert seq_camera_poses is not None
        assert seq_update_global is not None

        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
            seq_obstacle_locations=seq_obstacle_locations,
            seq_free_locations=seq_free_locations,
            blacklist_target=blacklist_target,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        map_features = seq_map_features.flatten(0, 1)
        # Compute the frontier map here
        frontier_map = self.policy.get_frontier_map(map_features)

        seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
            seq_found_goal[:, 0] == 0
        ]

        seq_goal_pose = None
        if len(all_matches) > 0 or matches is not None or self.instance_goal_found:
            (
                seq_goal_map,
                seq_found_goal,
                seq_goal_pose,
                self.instance_goal_found,
                self.goal_inst,
            ) = self.matching.select_and_localize_instance(
                seq_goal_map,
                seq_found_goal,
                final_local_map,
                seq_lmb[0],
                matches,
                confidence,
                local_instance_ids,
                stale_local_id_to_global_id_map,
                self.instance_goal_found,
                self.goal_inst,
                all_matches=all_matches,
                all_confidences=all_confidences,
                instance_ids=instance_ids,
                score_thresh=score_thresh,
            )

            seq_goal_map = seq_goal_map.view(
                batch_size, sequence_length, *seq_goal_map.shape[-2:]
            )

        else:
            # Predict high-level goals from map features
            # batched across sequence length x num environments
            if seq_object_goal_category is not None:
                seq_object_goal_category = seq_object_goal_category.flatten(0, 1)

            # Compute the goal map
            goal_map, found_goal = self.policy(
                map_features,
                seq_object_goal_category,
                reject_visited_targets=reject_visited_targets,
            )

            seq_goal_map = goal_map.view(
                batch_size, sequence_length, *goal_map.shape[-2:]
            )
            seq_found_goal = found_goal.view(batch_size, sequence_length)

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_goal_pose,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
    
class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        # self.max_steps = config.AGENT.max_steps
        self.max_steps = [500, 500, 500, 500, 500]
        # self.max_steps = [500, 400, 300, 200, 200, 200, 200, 200, 200, 200, 200]
        # self.max_steps = [400, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.panorama_rotate_steps = int(360 / config.ENVIRONMENT.turn_angle)

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self._module = GoatAgentModule(
            config, matching=self.matching, instance_memory=self.instance_memory
        )

        # if config.NO_GPU:
        self.device = torch.device("cpu")
        self.module = self._module
        # else:
        #     self.device_id = device_id
        #     self.device = torch.device(f"cuda:{self.device_id}")
        #     self._module = self._module.to(self.device)
        #     # Use DataParallel only as a wrapper to move model inputs to GPU
        #     self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        # self.imagenav_visualizer = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )
        self.goal_pose = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering
        self.prev_task_type = None

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
                the main inference function of the agent that lets it interact with
                vectorized environments.
        s
                This function assumes that the agent has been initialized.

                Args:
                    obs: current frame containing (RGB, depth, segmentation) of shape
                     (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
                    pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
                     of shape (num_environments, 3)
                    object_goal_category: semantic category of small object goals
                    camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

                Returns:
                    planner_inputs: list of num_environments planner inputs dicts containing
                        obstacle_map: (M, M) binary np.ndarray local obstacle map
                         prediction
                        sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                         and local map boundaries planning window (gx1, gx2, gy1, gy2)
                        goal_map: (M, M) binary np.ndarray denoting goal location
                    vis_inputs: list of num_environments visualization info dicts containing
                        explored_map: (M, M) binary np.ndarray local explored map
                         prediction
                        semantic_map: (M, M) np.ndarray containing local semantic map
                         predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if obstacle_locations is not None:
            obstacle_locations = obstacle_locations.unsqueeze(1)
        if free_locations is not None:
            free_locations = free_locations.unsqueeze(1)
        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        
        (
            self.goal_map,
            self.found_goal,
            self.goal_pose,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1),
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            seq_obstacle_locations=obstacle_locations,
            seq_free_locations=free_locations,
        )
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]


        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if self.found_goal[0].item():
            goal_map = self._prep_goal_map_input()

        # found_goal = self.found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            if frontier_map is not None:
                self.semantic_map.update_frontier_map(
                    e, frontier_map[e][0].cpu().numpy()
                )
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            self.sub_task_timesteps[e][self.current_task_idx] += 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )


        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
                "goal_pose": self.goal_pose[e] if self.goal_pose is not None else None
            }
            for e in range(self.num_environments)
        ]
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.total_timesteps[e],
                }
                for e in range(self.num_environments)
            ]
            if self.record_instance_ids:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )
        else:
            vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [
            [0] * self.max_num_sub_task_episodes
        ] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()
        self._module.reset()

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self._module.reset_sub_episode()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset_for_env(e)
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0
        self.planner.reset()
        self._module.reset()
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

    def score_thresh(self, task_type):
        # If we have fully explored the environment, set the matching threshold to 0.0
        # to go to the highest scoring instance
        if self.fully_explored[0]:
            return 0.0

        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]

        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            img_goal,
            camera_pose,
            keypoints,
            matches,
            confidence,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        ) = self._preprocess_obs(obs, task_type)

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            camera_pose=camera_pose,
            reject_visited_targets=self.reject_visited_targets,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
        )

        # t2 = time.time()
        # print(f"Mapping and goal selection: {t2 - t1:.2f}")

        # 3 - Planning
        closest_goal_map = None
        dilated_obstacle_map = None
        short_term_goal = None
        could_not_find_path = False
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.total_timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        else:
            (
                action,
                closest_goal_map,
                short_term_goal,
                dilated_obstacle_map,
                could_not_find_path,
                planner_stop
            ) = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.sub_task_timesteps[0][self.current_task_idx],
                debug=False
            )

        # t3 = time.time()
        # print(f"Planning: {t3 - t2:.2f}")

        if (
            self.sub_task_timesteps[0][self.current_task_idx]
            >= self.max_steps[self.current_task_idx]
        ):
            print("Reached max number of steps for subgoal, calling STOP")
            action = DiscreteNavigationAction.STOP

        if could_not_find_path and not planner_stop and action != DiscreteNavigationAction.STOP:
            # This doesn't help
            # print("Resetting explored area")
            # self.semantic_map.local_map[0, MC.EXPLORED_MAP] *= 0
            # self.semantic_map.global_map[0, MC.EXPLORED_MAP] *= 0

            # TODO: is this accurate?
            print("Can't find a path. Map fully explored.")
            self.fully_explored[0] = True
            self.force_match_against_memory = True

            # if self.reached_goal_candidate:
            #     # move to next sub-task
            #     # update semantic map
            #     # reset timesteps
            #     pass

        if self.visualize:
            if task_type == "imagenav":
                collision = {"is_collision": False}
                info = {
                    **planner_inputs[0],
                    **vis_inputs[0],
                    "rgb_frame": obs.rgb,
                    "depth_frame": obs.depth,
                    "semantic_frame": obs.task_observations["semantic_frame"],
                    "closest_goal_map": closest_goal_map,
                    "last_goal_image": obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                    "short_term_goal": short_term_goal,
                }
                if self.imagenav_visualizer is not None:
                    self.imagenav_visualizer.visualize(**info)
            else:
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                vis_inputs[0]["goal_name"] = goal_text_desc
                vis_inputs[0]["semantic_frame"] = obs.task_observations[
                    "semantic_frame"
                ]
                vis_inputs[0]["closest_goal_map"] = closest_goal_map
                vis_inputs[0]["third_person_image"] = obs.third_person_image
                vis_inputs[0]["short_term_goal"] = None
                vis_inputs[0]["instance_memory"] = self.instance_memory

                info = {
                    **planner_inputs[0],
                    **vis_inputs[0],
                    "short_term_goal": short_term_goal,
                }

        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.force_match_against_memory = False
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.reset_sub_episode()
        self.prev_task_type = task_type
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        current_task = obs.task_observations["tasks"][self.current_task_idx]
        current_goal_semantic_id = current_task["semantic_id"]

        semantic = obs.semantic
        instance_ids = None

        (
            matches,
            confidences,
            keypoints,
            local_instance_ids,
            all_matches,
            all_confidences,
            all_rgb_keypoints,
            instance_ids,
        ) = (None, None, None, None, [], [], [], [])

        if not self._module.instance_goal_found:
            if task_type == "imagenav":
                if self.goal_image is None:
                    img_goal = obs.task_observations["tasks"][self.current_task_idx][
                        "image"
                    ]
                    (
                        self.goal_image,
                        self.goal_image_keypoints,
                    ) = self.matching.get_goal_image_keypoints(img_goal)
                    # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.image_matching_function,
                    self.total_timesteps[0],
                    image_goal=self.goal_image,
                    goal_image_keypoints=self.goal_image_keypoints,
                    categories=[current_task["semantic_id"]],
                    use_full_image=False,
                )

            elif task_type == "languagenav":
                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.matching.match_language_to_image,
                    self.total_timesteps[0],
                    language_goal=current_task["description"],
                    categories=[current_task["semantic_id"]],
                    use_full_image=True,
                )
        
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose

        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0)

        # NOT USED AT ALL? ->
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        # Match a goal against every instance in memory the moment we get it
        # or when the map just got fully explored
        if (
            task_type in ["languagenav", "imagenav"]
            and self.record_instance_ids
            and (
                self.sub_task_timesteps[0][self.current_task_idx] == 0
                or self.force_match_against_memory
            )
        ):
            if self.force_match_against_memory:
                print("Force a match against the memory")
            self.force_match_against_memory = False
            (all_rgb_keypoints, all_matches, all_confidences, instance_ids) = self._match_against_memory(
                task_type, current_task
            )

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            self.goal_image,
            camera_pose,
            keypoints,
            matches,
            confidences,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        )

    def _match_against_memory(self, task_type: str, current_task: Dict):
        print("--------Matching against memory!--------")
        if task_type == "languagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.matching.match_language_to_image,
                self.total_timesteps[0],
                language_goal=current_task["description"],
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "all": scores.flatten().tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_language_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        elif task_type == "imagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.image_matching_function,
                self.sub_task_timesteps[0][self.current_task_idx],
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.sum(axis=1).mean()),
                    "median": float(np.median(scores.sum(axis=1))),
                    "max": float(scores.sum(axis=1).max()),
                    "min": float(scores.sum(axis=1).min()),
                    "all": scores.sum(axis=1).tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_image_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        return all_rgb_keypoints, all_matches, all_confidences, instance_ids

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map

        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue

            # cluster goal points
            try:
                c = DBSCAN(eps=4, min_samples=1)
                data = np.array(goal_map[e].nonzero()).T
                c.fit(data)

                # mask all points not in the largest cluster
                mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
                mode_mask = (c.labels_ != mode).nonzero()
                x = data[mode_mask]
                goal_map_ = np.copy(goal_map[e])
                goal_map_[x] = 0.0

                # adopt masked map if non-empty
                if goal_map_.sum() > 0:
                    goal_map[e] = goal_map_
            except Exception as e:
                print(e)
                return goal_map

        return goal_map