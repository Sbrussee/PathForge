from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathforge.inference.heatmaps import create_inference_heatmap
from pathforge.inference.model_package import (
    load_bag_from_input,
    load_packaged_model,
    select_inference_feature_metadata,
    predict_bag,
)

# Inference API should be small and stable


def main(argv: list[str] | None = None) -> int:
    """Run packaged-model inference and optional heatmap generation from the CLI."""
    p = argparse.ArgumentParser(description="Run PathForge-MIL inference utilities.")
    p.add_argument(
        "--model_path", required=True, help="Path to a trained model checkpoint."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Feature artifact path (.h5, .pt, .npy, or .npz).",
    )
    p.add_argument("--output", required=True, help="JSON prediction output path.")
    p.add_argument(
        "--feature-extractor",
        default=None,
        help="Optional H5 extractor key. Defaults to the packaged model metadata.",
    )
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
        "--slide-path",
        default=None,
        help="Optional source WSI path used to render full-resolution top-tile previews.",
    )
    p.add_argument(
        "--heatmap-image-output",
        default=None,
        help="Optional PNG path for a rendered heatmap preview image.",
    )
    args = p.parse_args(argv)

    loaded_package = load_packaged_model(args.model_path)
    resolved_bag_id, resolved_extractor = select_inference_feature_metadata(
        loaded_package,
        bag_id=args.bag_id,
        extractor_name=args.feature_extractor,
    )
    bag = load_bag_from_input(
        args.input,
        bag_id=resolved_bag_id,
        extractor_name=resolved_extractor,
    )
    predictions = predict_bag(
        loaded_package.model,
        bag,
        task=loaded_package.task,
    ).detach().cpu()
    res = _prediction_payload(
        predictions,
        task=loaded_package.task,
        model_name=loaded_package.model_name,
        bag_id=resolved_bag_id,
        feature_extractor=resolved_extractor,
    )
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
            slide_path=args.slide_path,
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
    return 0


def _prediction_payload(
    predictions,
    *,
    task: str,
    model_name: str,
    bag_id: str | None,
    feature_extractor: str | None,
) -> dict[str, object]:
    """Convert one normalized prediction tensor into a JSON-serializable payload."""

    prediction_list = predictions.squeeze(0).tolist()
    if not isinstance(prediction_list, list):
        prediction_list = [float(prediction_list)]
    payload: dict[str, object] = {
        "status": "ok",
        "task": task,
        "model_name": model_name,
        "predictions": prediction_list,
        "bag_id": bag_id,
        "feature_extractor": feature_extractor,
    }
    if task == "classification":
        import torch

        logits = predictions
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        payload["probs"] = probs
        payload["predicted_class"] = int(torch.argmax(logits, dim=-1).item())
    elif task in {"survival", "survival_discrete"}:
        payload["risk_scores"] = prediction_list
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
