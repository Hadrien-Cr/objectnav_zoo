import habitat_sim

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = (
    "data/scene_datasets/hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

import pprint

pprint.pprint(dir(sim_cfg))
