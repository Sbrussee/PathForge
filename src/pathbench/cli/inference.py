from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathbench.inference.heatmaps import create_inference_heatmap

# Inference API should be small and stable


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run PathBench-MIL inference utilities.")
    p.add_argument(
        "--model_path", required=True, help="Path to a trained model checkpoint."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Slide feature path, tile path, or slide H5 artifact.",
    )
    p.add_argument("--output", required=True, help="JSON prediction output path.")
    p.add_argument(
        "--heatmap-backend",
        default=None,
        help="Optional heatmap backend key. Use 'torchmil' for the TorchMIL heatmap explainer.",
    )
    p.add_argument(
        "--bag-id",
        default=None,
        help="Bag id for H5 coords, for example '256px_0.5mpp'.",
    )
    p.add_argument(
        "--scores",
        default=None,
        help="Optional .npy, .npz, or .json per-instance score vector shaped [N] for heatmap generation.",
    )
    p.add_argument(
        "--coords",
        default=None,
        help="Optional .npy, .npz, or .json coordinate matrix shaped [N,2]. If omitted, H5 bag coords are used.",
    )
    p.add_argument(
        "--mask",
        default=None,
        help="Optional .npy, .npz, or .json boolean/binary mask shaped [N] for heatmap generation.",
    )
    p.add_argument(
        "--heatmap-name",
        default="torchmil",
        help="H5 prediction heatmap namespace. Defaults to 'torchmil'.",
    )
    p.add_argument(
        "--heatmap-output",
        default=None,
        help="Optional JSON sidecar path for rendered heatmap coordinates and scores.",
    )
    p.add_argument(
        "--heatmap-image-output",
        default=None,
        help="Optional PNG path for a rendered heatmap preview image.",
    )
    args = p.parse_args(argv)

    # load model (Lightning checkpoint), run predict, save JSON output
    res = {"status": "ok", "probs": [0.1, 0.9]}
    if args.heatmap_backend is not None:
        if args.scores is None:
            raise ValueError("--scores is required when --heatmap-backend is set.")
        if args.bag_id is None:
            raise ValueError("--bag-id is required when --heatmap-backend is set.")
        heatmap_result = create_inference_heatmap(
            artifact_path=args.input,
            bag_id=args.bag_id,
            scores_path=args.scores,
            heatmap_backend=args.heatmap_backend,
            heatmap_name=args.heatmap_name,
            output_path=args.heatmap_output,
            image_output_path=args.heatmap_image_output,
            coords_path=args.coords,
            mask_path=args.mask,
            model_path=args.model_path,
        )
        res["heatmap"] = {
            "artifact_path": str(heatmap_result.artifact_path),
            "bag_id": heatmap_result.bag_id,
            "heatmap_name": heatmap_result.heatmap_name,
            "num_points": heatmap_result.num_points,
            "output_path": str(heatmap_result.output_path)
            if heatmap_result.output_path is not None
            else None,
            "image_output_path": (
                str(heatmap_result.image_output_path)
                if heatmap_result.image_output_path is not None
                else None
            ),
            "smoothed_image_output_path": (
                str(heatmap_result.smoothed_image_output_path)
                if heatmap_result.smoothed_image_output_path is not None
                else None
            ),
            "top_tiles_output_path": (
                str(heatmap_result.top_tiles_output_path)
                if heatmap_result.top_tiles_output_path is not None
                else None
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(res, f)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
