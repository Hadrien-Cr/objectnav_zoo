from typing import Any, Union, cast, Dict

import habitat
import os
import json
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import objectnav_zoo
from objectnav_zoo.perception.constants import (
    HM3DtoCOCOIndoor,
    LanguageNavCategories,
    all_hm3d_categories,
    coco_categories_mapping,
)
from objectnav_zoo.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from objectnav_zoo.env.habitat_abstract_env import HabitatEnv
from objectnav_zoo.env.habitat_goat_env.visualizer import Visualizer
from objectnav_zoo.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)

from objectnav_zoo.perception.constants import df as hm3d_mapping_df

HABITAT_DATA = os.environ["HABITAT_DATA"]

all_ovon_categories_path = HABITAT_DATA +  "/datasets/goat_bench/hm3d/v1/val_seen/goat_object_goals.json"

with open(all_ovon_categories_path, "r") as f:
    all_ovon_categories = json.load(f)

all_ovon_categories = ["_".join(x.split(" ")) for x in all_ovon_categories]


class HabitatGoatEnv(HabitatEnv):
    semantic_category_mapping: Union[HM3DtoCOCOIndoor]

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)

        self.episodes_data_path = config.habitat.dataset.data_path

        assert "hm3d" in self.episodes_data_path, "only HM3D scenes supported for now."

        if "hm3d" in self.episodes_data_path:
            self.semantic_category_mapping = LanguageNavCategories()

        self.config = config
        self.current_episode = None

        ovon_semantic_ids = []

        self.hm3d_mapping = {}

        for obj in self.habitat_env.sim.semantic_scene.objects:
            main_category = hm3d_mapping_df[hm3d_mapping_df['raw_category'] == obj.category.name()]
            
            # raw -> main category
            if len(main_category) == 0:
                continue
            else:
                main_category = main_category['category'].item()

            main_category = "_".join(main_category.split(" "))

            if main_category in all_ovon_categories:
                self.hm3d_mapping[int(obj.id.split('_')[-1])] = all_ovon_categories.index(main_category) + 1


        # for cat in hm3d_mapping_df['category'].tolist():
        #     if cat in all_ovon_categories:
        #         ovon_semantic_ids.append(all_ovon_categories.index(cat) + 1)
        #     else:
        #         ovon_semantic_ids.append(0)

        # hm3d_mapping_df['ovon_semantic_ids'] = ovon_semantic_ids

        # self.hm3d_mapping = hm3d_mapping_df.set_index('index')['ovon_semantic_ids'].to_dict()

        # self.segmentation = MaskRCNNPerception(
        #     sem_pred_prob_thr=0.9,
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        # )

        if not self.ground_truth_semantics:
            self.init_perception_module()

    def fetch_vocabulary(self, goals):
        # TODO: get open set vocabulary
        vocabulary = []
        for goal in goals:
            vocabulary.append(goal["target"])
            if "landmarks" in goal.keys():
                vocabulary += goal["landmarks"]
        return set(vocabulary)

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode
        self.active_task_idx = 0
        # goal_type, goal = self.update_and_fetch_goal()
        goals = habitat_obs["multigoal"]
        # open set vocabulary – all HM3D categories?
        # vocabulary = self.fetch_vocabulary(goals)
        # print("Vocabulary:", vocabulary)
        if not self.ground_truth_semantics:
            self.init_perception_module()

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
        self._last_obs = self._preprocess_obs(habitat_obs)
        self.visualizer.reset()
        scene_id = self.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[
            0
        ]
        self.visualizer.set_vis_dir(
            scene_id, self.habitat_env.current_episode.episode_id
        )

    def init_perception_module(self, vocabulary=None):
        from objectnav_zoo.perception.detection.detic.detic_perception import (
            DeticPerception,
        )

        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary="," + ",".join(all_ovon_categories),
            sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        )

        # self.segmentation = MaskRCNNPerception(
        #     sem_pred_prob_thr=0.9,
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        # )

        # from objectnav_zoo.perception.detection.grounded_sam.ram_perception import RAMPerception

        # self.segmentation = RAMPerception(
        #     custom_vocabulary=".",
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        #     verbose=False,
        #     # **module_kwargs
        # )


    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> objectnav_zoo.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])
        goals = self._preprocess_goals(
            self.current_episode.tasks, habitat_obs
        )

        agent_state = self.habitat_env._sim.get_agent_state()
        position = np.array(
            agent_state.position, dtype=np.float32
        )  # Ensure 1D array [x, y, z]
        rot_x, rot_y, rot_z, rot_w = (
            float(agent_state.rotation.x),
            float(agent_state.rotation.y),
            float(agent_state.rotation.z),
            float(agent_state.rotation.w),
        )
        quat = np.array([rot_w, rot_x, rot_y, rot_z], dtype=np.float32)
        from scipy.spatial.transform import Rotation as R

        rotation_matrix = R.from_quat(quat).as_matrix()  # 3x3 rotation matrix
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = position

        obs = objectnav_zoo.core.interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={
                "tasks": goals,
                "top_down_map": self.get_episode_metrics()["goat_top_down_map"],
            },
            camera_pose=camera_pose,
        )
        obs = self._preprocess_semantic(obs, habitat_obs["rgb"])
        return obs

    def _preprocess_semantic(
        self,
        obs: objectnav_zoo.core.interfaces.Observations,
        habitat_semantic: np.ndarray,
        vocabulary=None,
    ) -> objectnav_zoo.core.interfaces.Observations:
        if self.ground_truth_semantics:
            raise NotImplementedError
        else:
            obs = self.segmentation.predict(obs)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goals(self, tasks, habitat_obs):
        goals = []
        vocabulary = []

        goals = habitat_obs['multigoal']

        for goal_v in goals:
            goal_v["semantic_id"] = all_ovon_categories.index("_".join(goal_v["category"].split(" "))) + 1
            if goal_v["image"] is not None:
                goal_v["type"] = "imagenav"
            elif goal_v["description"] :
                goal_v["type"] = "languagenav"
            else:
                goal_v["type"] = "objectnav"

        return goals

    def _preprocess_action(self, action: objectnav_zoo.core.interfaces.Action) -> int:

        if type(action) == int:
            return action

        discrete_action = cast(
            objectnav_zoo.core.interfaces.DiscreteNavigationAction, action
        )
        return HabitatSimActions[discrete_action.name.lower()]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            if (
                self.habitat_env.current_episode.tasks[
                    self.habitat_env.task.current_task_idx
                ][1]
                != "image"
            ):
                info["top_down_map"] = self.get_observation().task_observations.get("top_down_map")
                self.visualizer.visualize(**info)