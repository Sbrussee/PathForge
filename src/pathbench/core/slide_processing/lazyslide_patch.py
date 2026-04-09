# myrepo/patches/lazyslide_feature_extraction_patch.py

from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_features

import lazyslide as zs
import lazyslide._api as _api
import lazyslide.tools._features as _features
from lazyslide._const import Key
from lazyslide._utils import default_pbar
from lazyslide.models import ImageModel


def _restore_image_like_tile(tile):
    """Convert color-normalized output back to the image-like format transforms expect."""
    if isinstance(tile, torch.Tensor):
        if tile.ndim == 3 and tile.shape[0] in (1, 3, 4) and tile.shape[-1] not in (1, 3, 4):
            tile = tile.permute(1, 2, 0)
        tile = tile.detach().cpu()
        if tile.dtype.is_floating_point:
            tile = tile.clamp(0, 255).round()
        return tile.to(torch.uint8).numpy()

    if isinstance(tile, np.ndarray):
        if tile.ndim == 3 and tile.shape[0] in (1, 3, 4) and tile.shape[-1] not in (1, 3, 4):
            tile = np.transpose(tile, (1, 2, 0))
        if np.issubdtype(tile.dtype, np.floating):
            tile = np.clip(tile, 0, 255).round().astype(np.uint8)
        return tile

    return tile


def feature_extraction_patched(
    wsi: WSIData,
    model: str | Callable | ImageModel = None,
    model_path: str | Path = None,
    model_name: str = None,
    jit: bool = False,
    token: str = None,
    load_kws: dict = None,
    transform: Callable = None,
    color_norm: str = None,   # <- added
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    tile_key: str = Key.tiles,
    key_added: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pbar: bool = None,
    return_features: bool = False,
    **kwargs,
):
    device = _api.default_value("device", device)
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)
    pbar = _api.default_value("pbar", pbar)

    load_kws = {} if load_kws is None else load_kws

    if model is not None:
        if isinstance(model, Callable):
            model = model
        elif isinstance(model, str):
            model, default_model_name = _features.load_models(
                model_name=model,
                model_path=model_path,
                token=token,
                **load_kws,
            )
            if model_name is None:
                model_name = default_model_name
        elif isinstance(model, ImageModel):
            model = model
            model_name = model.name
        else:
            raise ValueError("Model must be a model name or a model object.")
    else:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        model_path = Path(model_path)
        if model_path.exists():
            load_kws.setdefault("weights_only", False)
            load_func = torch.load if not jit else torch.jit.load
            model = load_func(model_path, **load_kws)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    if key_added is None:
        if model_name is not None:
            key_added = model_name
        elif isinstance(model, ImageModel):
            key_added = model.name
        elif hasattr(model, "__class__"):
            key_added = model.__class__.__name__
        elif hasattr(model, "__name__"):
            key_added = model.__name__
        else:
            key_added = "features"
        key_added = Key.feature(key_added, tile_key)

    try:
        model.to(device)
    except Exception:
        pass

    if transform is None and isinstance(model, ImageModel):
        transform = model.get_transform()

    if color_norm is not None and transform is not None:
        base_transform = transform

        def transform(tile):
            return base_transform(_restore_image_like_tile(tile))

    n_tiles = len(wsi.shapes[tile_key])

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=n_tiles)

        dataset = wsi.ds.tile_images(
            tile_key=tile_key,
            transform=transform,
            color_norm=color_norm,   # <- only real behavioral change
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        features = []
        if isinstance(device, torch.device):
            device = device.type

        amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()

        with amp_ctx, torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device)
                if isinstance(model, ImageModel):
                    output = model.encode_image(image)
                else:
                    output = model(image)

                if not isinstance(output, np.ndarray):
                    output = output.cpu().numpy()

                features.append(output)
                progress_bar.update(task, advance=len(image))
                del batch

        progress_bar.refresh()
        features = np.vstack(features)

    add_features(wsi, key=key_added, tile_key=tile_key, features=features)

    if return_features:
        return features
    return None


def apply_lazyslide_feature_extraction_patch():
    zs.tl.feature_extraction = feature_extraction_patched
    _features.feature_extraction = feature_extraction_patched
