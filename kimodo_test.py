import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = r"kimodo\saved_data\fist_bump.npz"
save_img_path= r"kimodo\saved_images"

data = np.load(file_path)

for key in data.files:
    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")

joints = data["posed_joints"]       # (180, 77, 3)
root = data["root_positions"]       # (180, 3)
contacts = data["foot_contacts"]    # (180, 6)
rot_mats = data["global_rot_mats"]  # (180, 77, 3, 3)

T, J, _ = joints.shape
print(f"Frames: {T}, Joints: {J}, Duration: {T/30:.1f}s")

# ── 1. Single-frame skeleton (first, middle, last) ──
fig = plt.figure(figsize=(18, 6))
for i, frame_idx in enumerate([0, T//2, T-1]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    pos = joints[frame_idx]  # (77, 3)
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=8, c='blue', alpha=0.7)
    # highlight root
    ax.scatter(root[frame_idx, 0], root[frame_idx, 2], root[frame_idx, 1],
               s=60, c='red', marker='x', label='root')
    ax.set_title(f"Frame {frame_idx} ({frame_idx/30:.2f}s)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.legend()
    # set equal aspect ratio
    max_range = (pos.max(axis=0) - pos.min(axis=0)).max() / 2
    mid = pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[2] - max_range, mid[2] + max_range)
    ax.set_zlim(mid[1] - max_range, mid[1] + max_range)
plt.suptitle("Skeleton Poses at Three Timepoints")
plt.tight_layout()
plt.savefig(os.path.join(save_img_path, "01_skeleton_poses.png"), dpi=150)
plt.show()

# ── 2. Root trajectory (bird's eye view) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# top-down (X vs Z)
axes[0].plot(root[:, 0], root[:, 2], 'b-', linewidth=1.5)
axes[0].plot(root[0, 0], root[0, 2], 'go', markersize=10, label='start')
axes[0].plot(root[-1, 0], root[-1, 2], 'ro', markersize=10, label='end')
axes[0].set_xlabel("X")
axes[0].set_ylabel("Z")
axes[0].set_title("Root Path (Top-Down)")
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].grid(True)

# height over time
axes[1].plot(np.arange(T) / 30, root[:, 1], 'b-', linewidth=1.5)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Y Height (m)")
axes[1].set_title("Root Height Over Time")
axes[1].grid(True)

# all three coordinates over time
time = np.arange(T) / 30
for dim, label, color in zip([0, 1, 2], ['X', 'Y (up)', 'Z'], ['r', 'g', 'b']):
    axes[2].plot(time, root[:, dim], color=color, label=label, linewidth=1.5)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Position (m)")
axes[2].set_title("Root Position Components")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_img_path, "02_root_trajectory.png"), dpi=150)
plt.show()

# ── 3. Foot contacts over time ──
fig, ax = plt.subplots(figsize=(14, 4))
contact_labels = [f"Contact {i}" for i in range(contacts.shape[1])]
for i in range(contacts.shape[1]):
    ax.fill_between(time, contacts[:, i] * (i + 1), i, alpha=0.6, label=contact_labels[i])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Contact Channel")
ax.set_title("Foot Contacts Over Time (filled = contact)")
ax.set_yticks(range(contacts.shape[1]))
ax.set_yticklabels(contact_labels)
ax.grid(True, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(save_img_path, "03_foot_contacts.png"), dpi=150)
plt.show()

# ── 4. Motion trail (all frames overlaid) ──
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# plot every 6th frame (~5fps) for clarity
for frame_idx in range(0, T, 6):
    pos = joints[frame_idx]
    alpha = 0.2 + 0.8 * (frame_idx / T)  # fade in over time
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], s=2, c='blue', alpha=alpha)
# root trajectory
ax.plot(root[:, 0], root[:, 2], root[:, 1], 'r-', linewidth=2, label='root path')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y (up)")
ax.set_title("Motion Trail (every 6th frame)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_img_path, "04_motion_trail.png"), dpi=150)
plt.show()

# ── 5. Joint velocity magnitudes ──
joint_velocities = np.diff(joints, axis=0) * 30  # convert to m/s
velocity_magnitudes = np.linalg.norm(joint_velocities, axis=-1)  # (T-1, 77)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(velocity_magnitudes.T, aspect='auto', cmap='hot',
               extent=[0, (T-1)/30, J-0.5, -0.5])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Joint Index")
ax.set_title("Joint Velocity Magnitudes (m/s)")
plt.colorbar(im, ax=ax, label="Speed (m/s)")
plt.tight_layout()
plt.savefig(os.path.join(save_img_path, "05_joint_velocities.png"), dpi=150)
plt.show()

print("All plots saved.")