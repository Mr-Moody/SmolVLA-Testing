"""
SmolVLA setup for Franka Panda 7.

Downloads lerobot/smolvla_base and demonstrates the expected input/output
data format for a 7-DOF Franka arm with one wrist camera.

Run from the lerobot repo with:
    uv run python ../SmolVLA-Testing/smolvla_franka_setup.py

Dependencies (add smolvla extra first):
    uv sync --locked --extra training --extra hardware --extra dataset --extra smolvla
"""

import torch
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors

MODEL_ID = "lerobot/smolvla_base"

# Franka Panda 7: 7 joint positions + 1 gripper = 8 dims
# SmolVLA pads states/actions to max_state_dim (32) internally.
FRANKA_STATE_DIM = 8   # q1..q7 + gripper
FRANKA_ACTION_DIM = 8  # target q1..q7 + gripper command

TASK = "pick up the red block and place it in the bin"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading SmolVLA from '{MODEL_ID}' ...")
policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
policy = policy.to(DEVICE)
policy.eval()

print(f"\nModel config:")
print(f"  VLM backbone : {policy.config.vlm_model_name}")
print(f"  chunk_size   : {policy.config.chunk_size}  (actions predicted per inference call)")
print(f"  n_action_steps: {policy.config.n_action_steps}  (actions executed before re-querying)")
print(f"  max_state_dim : {policy.config.max_state_dim}  (Franka 8D will be zero-padded to this)")
print(f"  max_action_dim: {policy.config.max_action_dim}")
print(f"  Image resize  : {policy.config.resize_imgs_with_padding}")


# The preprocessor tokenises the task string and normalises observations.
# The postprocessor denormalises the action back to robot units.
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
)

# In real use these values come from:
#   - Joint states : franka_ros /franka_state_controller/joint_states
#   - Camera image : /camera/color/image_raw  (converted to float32 [0,1])
#
# Key names must match the keys in the dataset the model was fine-tuned on.
# For the base model the expected camera key is "observation.images.top"
# (check the model card on the Hub for exact keys).

batch_size = 1
H, W = 480, 640  # raw camera resolution (policy resizes internally to 512x512)

raw_obs = {
    # RGB image from the wrist/top camera: float32, range [0, 1], shape (B, C, H, W)
    "observation.images.top": torch.rand(batch_size, 3, H, W, device=DEVICE),

    # Joint state vector: 7 joint angles (rad) + 1 gripper width (m)
    # Shape (B, state_dim).  Values below are illustrative home-pose angles.
    "observation.state": torch.tensor(
        [[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04]],
        dtype=torch.float32,
        device=DEVICE,
    ),
}

print(f"\nRaw observation keys and shapes:")
for k, v in raw_obs.items():
    print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")


processed_obs = preprocess({**raw_obs, "task": TASK})

print(f"\nProcessed observation keys and shapes:")
for k, v in processed_obs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} = {v!r}")

# select_action returns one action step at a time (shape: [B, action_dim]).
# Internally it calls the model every n_action_steps and caches the rest.

policy.reset()  # clear action queue between episodes

action_tensor = policy.select_action(processed_obs)

print(f"\nRaw action tensor shape : {tuple(action_tensor.shape)}")
print(f"  (batch={action_tensor.shape[0]}, action_dim={action_tensor.shape[1]})")

# Denormalises back from model space to robot units.
action_denorm = postprocess(action_tensor)

print(f"\nDenormalised action shape: {tuple(action_denorm.shape)}")

# Trim to Franka's actual DoF
franka_action = action_denorm[0, :FRANKA_ACTION_DIM].cpu()

print(f"\nFranka action (first step):")
print(f"  Joint targets (q1-q7) [rad] : {franka_action[:7].tolist()}")
print(f"  Gripper command       [m]   : {franka_action[7].item():.4f}")
print(f"  (positive = open, 0 = closed, max ~0.08 m)")


# Send joint targets to the robot via franka_ros:
#
#   rostopic pub /franka_control/joint_position_command \
#       franka_msgs/JointPositions \
#       "q: [q1, q2, q3, q4, q5, q6, q7]"
#
# Or via franka_ros Python:
#   from franka_msgs.msg import JointPositions
#   pub.publish(JointPositions(q=franka_action[:7].tolist()))
#
# The gripper is controlled separately:
#   rosaction send /franka_gripper/move franka_gripper/MoveAction \
#       "width: 0.04, speed: 0.1"
