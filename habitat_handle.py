import habitat_sim
import numpy as np
from magnum import mesh_tools, trade

# 1. Load the ReplicaCAD configuration (update the path)
sim_settings = {
    "scene_dataset_config_file": "/path/to/data/replica_cad/replicaCAD.scene_dataset_config.json",
    "scene_id": "apt_1"  # or any other scene ID
}
cfg = habitat_sim.Configuration(
    habitat_sim.SimulatorConfiguration()
)
cfg.sim_cfg.scene_dataset_config_file = sim_settings["scene_dataset_config_file"]
cfg.sim_cfg.load_semantic_mesh = True  # Loads the mesh we want to export
sim = habitat_sim.Simulator(cfg)

# 2. Get the scene graph and the root node
scene_graph = sim.get_active_scene_graph()
root_node = scene_graph.get_root_node()

# 3. Find the "stage" node (where the static mesh usually is)
stage_node = None
def find_stage_node(node):
    global stage_node
    if node.node_type == habitat_sim.SceneNodeType.STAGE:
        stage_node = node
        return True
    for child in node.children:
        if find_stage_node(child):
            return True
    return False
find_stage_node(root_node)

if stage_node is None:
    print("Stage node not found. Check the scene structure.")
    # Fallback: try to get the first drawable mesh from the simulator
    rigid_states = sim.get_rigid_object_manager().get_objects_by_handle_substring("stage")
    if rigid_states:
        # For a more complex export, you would need to access the mesh data via the RenderAssetInstanceCreator
        pass
else:
    # 4. Export the mesh (conceptual - may need adjustments)
    # Access the mesh data from the node's visual component
    # The exact method depends on the Habitat-sim version
    # You may need to use `mesh_tools.combine_meshes` if the mesh is split into multiple parts
    # mesh = stage_node.visuals[0].mesh
    # mesh_data = mesh_tools.export_to_ply(mesh)
    # with open("output.ply", "wb") as f:
    #     f.write(mesh_data)
    print(f"Found stage node: {stage_node.id}")

    # Alternatively, try to get the semantic mesh
    semantic_mesh = sim.semantic_scene
    if semantic_mesh:
        # Access the mesh data via semantic_mesh
        print(f"Semantic mesh loaded with {len(semantic_mesh.objects)} objects")
        # You would need to iterate through semantic objects and combine their meshes
        pass

sim.close()