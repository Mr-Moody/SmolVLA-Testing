"""
Patches lerobot's encode_video_frames to use the system ffmpeg binary when
vcodec='h264_nvenc' is requested, bypassing PyAV's bundled ffmpeg 7.x which
requires NVENC API 13.0. The system ffmpeg on UCL TSG trailbreaker (Lavc59,
ffmpeg 5.x) works with the installed NVENC 12.2 driver.

Safe to re-run — removes any previously applied version of the patch before
inserting the current one.
"""
from pathlib import Path

TARGET = Path.home() / "smolvla_project/lerobot/src/lerobot/datasets/video_utils.py"

# Marker that identifies any version of our patch
PATCH_MARKER = "# h264_nvenc: PyAV 15.x bundles ffmpeg 7.x which requires NVENC API 13.0."

# The insertion anchor that exists in the unpatched file
ANCHOR = "    # Define video codec options\n    video_options = _get_codec_options(vcodec, g, crf, preset)"

# The full patch block to insert (includes the anchor at the end so we replace it)
PATCH = """\
    # h264_nvenc: PyAV 15.x bundles ffmpeg 7.x which requires NVENC API 13.0.
    # The UCL TSG trailbreaker driver only exposes NVENC 12.2, causing PyAV to
    # fail at codec open time. The system ffmpeg (Lavc59, ffmpeg 5.x) was
    # compiled against an older NVENC SDK and works fine with this driver.
    # We shell out to system ffmpeg for this codec and return early.
    # -pix_fmt yuv420p is required: h264_nvenc defaults to nv12 which
    # lerobot's get_video_pixel_channels does not recognise.
    if vcodec == "h264_nvenc":
        import subprocess as _sp
        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-y" if overwrite else "-n",
            "-framerate", str(fps),
            "-i", str(imgs_dir / "frame-%06d.png"),
            "-c:v", "h264_nvenc",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        result = _sp.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"System ffmpeg h264_nvenc encoding failed.\\n"
                f"Command: {' '.join(cmd)}\\n"
                f"Stderr: {result.stderr}"
            )
        return

    # Define video codec options
    video_options = _get_codec_options(vcodec, g, crf, preset)"""

source = TARGET.read_text()

# Remove any previously applied version of the patch so we can re-insert cleanly
if PATCH_MARKER in source:
    print("Previous patch found — removing it before re-applying...")
    # Find the start of our patch block
    start = source.index(PATCH_MARKER)
    # Walk back to the start of the indented block (the line before PATCH_MARKER)
    block_start = source.rfind("\n", 0, start) + 1
    # Find the end: the anchor line always follows our block
    end_marker = "    # Define video codec options\n    video_options = _get_codec_options(vcodec, g, crf, preset)"
    end = source.index(end_marker) + len(end_marker)
    source = source[:block_start] + end_marker + source[end:]
    print("Old patch removed.")

if ANCHOR not in source:
    print("ERROR: Could not find insertion point in video_utils.py.")
    print("Look for this line manually:")
    print("    video_options = _get_codec_options(vcodec, g, crf, preset)")
else:
    patched = source.replace(ANCHOR, PATCH, 1)
    TARGET.write_text(patched)
    print(f"Patched successfully: {TARGET}")
    print("encode_video_frames now uses system ffmpeg for h264_nvenc with -pix_fmt yuv420p.")
