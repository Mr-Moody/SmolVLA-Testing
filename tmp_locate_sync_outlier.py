import json
from bisect import bisect_left
from pathlib import Path

base = Path("raw_recordings/003")
robot = [json.loads(line) for line in (base / "robot.jsonl").open() if line.strip()]
events = [json.loads(line) for line in (base / "episode_events.jsonl").open() if line.strip()]
frames = [json.loads(line) for line in (base / "cameras/ee_zed_m/frames.jsonl").open() if line.strip()]

def ts(row):
    for key in ("host_timestamp_ns", "timestamp_ns", "robot_timestamp_ns", "receive_host_time_ns", "created_unix_time_ns"):
        if row.get(key) is not None:
            return int(row[key])
    raise KeyError("no ts")

robot_ts = [ts(r) for r in robot]
frame_ts = [ts(f) for f in frames]
starts = sorted(int(e["robot_timestamp_ns"]) for e in events if e.get("event") == "episode_start" and e.get("robot_timestamp_ns") is not None)

# target from converter error
target_episode = 31
target_step = 1563

s = starts[target_episode]
n = starts[target_episode + 1] if target_episode + 1 < len(starts) else None
si = bisect_left(robot_ts, s)
ei = bisect_left(robot_ts, n) if n is not None else len(robot)

global_idx = si + target_step
rts = robot_ts[global_idx]

insert = bisect_left(frame_ts, rts)
cands = []
if insert < len(frame_ts):
    cands.append((abs(frame_ts[insert] - rts), insert, frame_ts[insert]))
if insert > 0:
    i2 = insert - 1
    cands.append((abs(frame_ts[i2] - rts), i2, frame_ts[i2]))

cands.sort(key=lambda x: x[0])

print("episode_start_idx", si, "episode_end_idx", ei, "episode_len", ei - si)
print("target_global_idx", global_idx)
print("target_robot_ts", rts)
for d, fi, fts in cands[:2]:
    print("nearest_frame_idx", fi, "frame_ts", fts, "delta_ms", d / 1e6)

for off in range(-3, 4):
    gi = global_idx + off
    if 0 <= gi < len(robot_ts):
        print("neighbor", off, gi, robot_ts[gi])
