"""
check_cameras.py — Preflight camera validation for training runs.

Verifies that every dataset in a run has all required cameras with rgb.mp4
present before cleaning, converting, or training begins.

Usage:
    python src/check_cameras.py <dataset_name> [<dataset_name2> ...] [--cameras CAM1,CAM2]
    python src/check_cameras.py 200 201 202 203 204 201_1

Raises SystemExit(1) on any missing camera or video file.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_RAW_ROOT = Path("raw_datasets")
DEFAULT_CAMERAS = ["wrist_d405", "third_person_d405"]


def check_dataset_cameras(
    dataset_name: str,
    raw_root: Path,
    required_cameras: list[str],
) -> list[str]:
    """Return a list of error strings for a single dataset. Empty = OK."""
    errors = []
    ds_dir = raw_root / dataset_name
    if not ds_dir.exists():
        errors.append(f"[{dataset_name}] dataset directory not found: {ds_dir}")
        return errors

    cameras_dir = ds_dir / "cameras"
    if not cameras_dir.exists():
        errors.append(f"[{dataset_name}] cameras/ directory missing in {ds_dir}")
        return errors

    for cam in required_cameras:
        cam_dir = cameras_dir / cam
        if not cam_dir.exists():
            errors.append(f"[{dataset_name}] camera directory missing: cameras/{cam}/")
            continue
        rgb = cam_dir / "rgb.mp4"
        if not rgb.exists():
            errors.append(
                f"[{dataset_name}] rgb.mp4 missing for camera '{cam}': {rgb}\n"
                f"  Hint: check the recording saved video for this camera."
            )

    return errors


def check_all(
    dataset_names: list[str],
    raw_root: Path,
    required_cameras: list[str],
) -> bool:
    """Check all datasets. Prints results and returns True if all pass."""
    all_errors = []
    for name in dataset_names:
        errors = check_dataset_cameras(name, raw_root, required_cameras)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"  [{name}] OK — {', '.join(required_cameras)}")

    if all_errors:
        print("\nERROR: Camera preflight check FAILED:", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(
            "\nDo not proceed with training until all cameras have rgb.mp4 files.",
            file=sys.stderr,
        )
        return False

    print(f"\nAll {len(dataset_names)} datasets passed camera preflight check.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight camera check — verify all datasets have required camera video files."
    )
    parser.add_argument("dataset_names", nargs="+", help="Dataset names under --raw-root.")
    parser.add_argument(
        "--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
        help=f"Root containing raw datasets (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--cameras", type=str, default=",".join(DEFAULT_CAMERAS),
        help=f"Comma-separated required camera names (default: {','.join(DEFAULT_CAMERAS)}).",
    )
    args = parser.parse_args()
    required_cameras = [c.strip() for c in args.cameras.split(",")]

    print(f"Checking cameras: {required_cameras}")
    print(f"Datasets: {args.dataset_names}\n")

    ok = check_all(args.dataset_names, args.raw_root, required_cameras)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
