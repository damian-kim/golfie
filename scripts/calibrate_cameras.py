#!/usr/bin/env python3
"""CLI script to perform intrinsic and stereo camera calibration.

Usage:
    python scripts/calibrate_cameras.py \
      --images-a data/calibration/cam_a/ \
      --images-b data/calibration/cam_b/ \
      --shared-a data/calibration/shared_a/ \
      --shared-b data/calibration/shared_b/ \
      --board-type charuco \
      -o data/calibration/calibration_result.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import cv2

from golfie_cv.calibration import calibrate_intrinsics, calibrate_stereo
from golfie_core.schemas import CalibrationResult


def extract_frames_from_video(video_path: Path, max_frames: int = 30) -> List[Path]:
    """Helper to extract up to `max_frames` evenly spaced frames from a video
    and save them as temporary PNGs in the same directory.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // max_frames)
    frame_paths = []
    
    tmp_dir = video_path.parent / f"_tmp_frames_{video_path.stem}"
    tmp_dir.mkdir(exist_ok=True)

    frame_idx = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0 and saved_count < max_frames:
            out_path = tmp_dir / f"frame_{saved_count:04d}.png"
            cv2.imwrite(str(out_path), frame)
            frame_paths.append(out_path)
            saved_count += 1
            
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path.name} to {tmp_dir}")
    return frame_paths


def resolve_image_paths(paths_or_dirs: List[str]) -> List[Path]:
    """Resolve a list of files, directories, or videos to a list of image paths."""
    resolved = []
    for item in paths_or_dirs:
        p = Path(item)
        if not p.exists():
            print(f"Warning: path {item} does not exist. Skipping.")
            continue
            
        if p.is_dir():
            # Gather common image extensions
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                resolved.extend(p.glob(ext))
        elif p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
            resolved.extend(extract_frames_from_video(p))
        else:
            resolved.append(p)
            
    # Sort paths for consistency
    resolved.sort()
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Golfie Camera Calibration Script (Milestone 1)"
    )
    parser.add_argument(
        "--images-a",
        nargs="+",
        required=True,
        help="Images, video, or directories containing calibration patterns for Camera A intrinsic calibration.",
    )
    parser.add_argument(
        "--images-b",
        nargs="+",
        required=True,
        help="Images, video, or directories containing calibration patterns for Camera B intrinsic calibration.",
    )
    parser.add_argument(
        "--shared-a",
        nargs="+",
        required=True,
        help="Images, video, or directories containing calibration patterns visible in Camera A for stereo calibration.",
    )
    parser.add_argument(
        "--shared-b",
        nargs="+",
        required=True,
        help="Images, video, or directories containing calibration patterns visible in Camera B for stereo calibration.",
    )
    parser.add_argument(
        "--board-type",
        choices=["chessboard", "charuco"],
        default="charuco",
        help="Type of calibration target used.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=11,
        help="Number of square columns on the board.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=8,
        help="Number of square rows on the board.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.04,
        help="Square size in meters.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.03,
        help="ChArUco marker size in meters.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="calibration_result.json",
        help="Output JSON file path for CalibrationResult.",
    )

    args = parser.parse_args()

    print("==> Resolving image paths...")
    imgs_a = resolve_image_paths(args.images_a)
    imgs_b = resolve_image_paths(args.images_b)
    shared_a = resolve_image_paths(args.shared_a)
    shared_b = resolve_image_paths(args.shared_b)

    print(f"Found {len(imgs_a)} calibration frames for Camera A.")
    print(f"Found {len(imgs_b)} calibration frames for Camera B.")
    print(f"Found {len(shared_a)} shared stereo frames for Camera A.")
    print(f"Found {len(shared_b)} shared stereo frames for Camera B.")

    print("\n==> Calibrating Camera A Intrinsics...")
    intrinsics_a = calibrate_intrinsics(
        imgs_a,
        board_type=args.board_type,
        grid_size=(args.grid_cols, args.grid_rows),
        square_length=args.square_size,
        marker_length=args.marker_size,
    )
    print(f"Camera A Intrinsics calibrated successfully. Reprojection error: {intrinsics_a.reprojection_error_px:.4f} px")

    print("\n==> Calibrating Camera B Intrinsics...")
    intrinsics_b = calibrate_intrinsics(
        imgs_b,
        board_type=args.board_type,
        grid_size=(args.grid_cols, args.grid_rows),
        square_length=args.square_size,
        marker_length=args.marker_size,
    )
    print(f"Camera B Intrinsics calibrated successfully. Reprojection error: {intrinsics_b.reprojection_error_px:.4f} px")

    print("\n==> Performing Stereo Extrinsic Calibration...")
    stereo_result = calibrate_stereo(
        intrinsics_a,
        intrinsics_b,
        shared_a,
        shared_b,
        board_type=args.board_type,
        grid_size=(args.grid_cols, args.grid_rows),
        square_length=args.square_size,
        marker_length=args.marker_size,
    )
    print(f"Stereo Extrinsic Calibration completed. Reprojection error: {stereo_result.reprojection_error_px:.4f} px")
    print(f"Confidence score: {stereo_result.confidence:.4f}")

    # Write output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stereo_result.model_dump_json(indent=2))
    print(f"\nSaved CalibrationResult to {output_path.resolve()}")


if __name__ == "__main__":
    main()
