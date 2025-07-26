import motrixsim as mtx
import time
from scipy.spatial.transform import Rotation
import numpy as np
import math

model = mtx.load_model("model.xml")
data = mtx.SceneData(model)
mtx.forward_kinematic(model, data)

render = mtx.render.RenderApp()
render.launch(model)

top_plat = model.get_link("top")
top_plat_pos_init = top_plat.get_pose(data)[0:3]  # initial position of top plat

# world space positions of bottom connects
bottom_connect_pos = []
# initial leg lengths
leg_length_init = []
# offsets of top connects from top plat center
top_connect_offset = []

for i in range(3):
    for j in range(2):
        top_connect = model.get_link(f"top_connect{j}{i}")
        leg = model.get_link(f"leg{j}{i}")
        bottom_connect_pos.append(leg.get_pose(data)[0:3])
        top_connect_pos = top_connect.get_pose(data)[0:3]
        leg_length_init.append(np.linalg.norm(top_connect_pos - bottom_connect_pos[-1]))
        top_connect_offset.append(top_connect_pos - top_plat_pos_init)


target_pos = top_plat_pos_init
target_rot = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)


def compute_leg_joint_pos(i: int, target_pos, target_rot):
    expected_connect_pos = target_pos + target_rot.apply(top_connect_offset[i])
    return (
        np.linalg.norm(expected_connect_pos - bottom_connect_pos[i])
        - leg_length_init[i]
    )


actuators = [model.get_actuator(f"a{i}") for i in range(6)]

# number of physics steps per render step
phy_steps = math.ceil(0.02 / model.options.timestep)


target_pos_min = target_pos - np.array([0.5, 0.5, 0.5])
target_pos_max = target_pos + np.array([0.5, 0.5, 0.5])


while True:
    time.sleep(0.02)

    for i in range(6):
        q = compute_leg_joint_pos(i, target_pos, target_rot)
        actuators[i].set_ctrl(data, q)

    for i in range(phy_steps):
        mtx.step(model, data)

    render.sync(data)

    input = render.input

    move_speed = 0.04

    rotate_speed = 2  # degrees per step
    if input.is_key_pressed("left"):
        target_pos[0] -= move_speed
    if input.is_key_pressed("right"):
        target_pos[0] += move_speed
    if input.is_key_pressed("up"):
        target_pos[1] += move_speed
    if input.is_key_pressed("down"):
        target_pos[1] -= move_speed
    if input.is_key_pressed("w"):
        target_pos[2] += move_speed
    if input.is_key_pressed("s"):
        target_pos[2] -= move_speed

    target_pos = np.clip(target_pos, target_pos_min, target_pos_max)

    if input.is_key_pressed("a"):
        target_rot = (
            Rotation.from_rotvec(np.array([0, 0, rotate_speed]), degrees=True)
            * target_rot
        )
    if input.is_key_pressed("d"):
        target_rot = (
            Rotation.from_rotvec(np.array([0, 0, -rotate_speed]), degrees=True)
            * target_rot
        )
    if input.is_key_pressed("q"):
        target_rot = (
            Rotation.from_rotvec(np.array([0, rotate_speed, 0]), degrees=True)
            * target_rot
        )
    if input.is_key_pressed("e"):
        target_rot = (
            Rotation.from_rotvec(np.array([0, -rotate_speed, 0]), degrees=True)
            * target_rot
        )
    if input.is_key_pressed("z"):
        target_rot = (
            Rotation.from_rotvec(np.array([rotate_speed, 0, 0]), degrees=True)
            * target_rot
        )
    if input.is_key_pressed("c"):
        target_rot = (
            Rotation.from_rotvec(np.array([-rotate_speed, 0, 0]), degrees=True)
            * target_rot
        )
