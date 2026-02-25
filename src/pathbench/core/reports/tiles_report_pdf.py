# src/pathbench/core/reporting/tiles_report_pdf.py
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import tiles as tiles_io

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TilesOverviewEntry:
    slide_id: str
    artifact_path: Path
    image_bytes: bytes
    num_tiles: int | None
    tiling_spec: dict[str, Any] | None


@dataclass(slots=True)
class TilesReportStats:
    total_slides_expected: int
    included_slides: int
    missing_overview: int
    missing_coords: int
    unreadable_h5: int
    corrupt_overview: int


@dataclass(slots=True)
class TilesReportCollection:
    entries: list[TilesOverviewEntry]
    stats: TilesReportStats
    bag_id: str
    representative_tiling_spec: dict[str, Any] | None


def create_tiles_report_pdf(
    *,
    dataset: WSIDataset,
    bag_id: str,
    output_path: Path | None = None,
    timestamp: datetime | None = None,
    page_size: tuple[float, float] = A4,
) -> Path:
    """
    Create a timestamped tile extraction report PDF for one dataset and one bag_id.

    The report is built from H5-stored `tiles_overview` images, so original WSI files are
    not required.

    Args:
        dataset: PathBench WSIDataset (provides slide list/order and artifact paths).
        bag_id: Tiling bag identifier, e.g. "256px_0.5mpp".
        output_path: Optional explicit output PDF path. If None, a timestamped file is
            created in dataset.artifacts_dir.
        timestamp: Optional datetime used for deterministic naming/testing.
        page_size: ReportLab page size (default A4 portrait).

    Returns:
        Path to the written PDF.

    Raises:
        RuntimeError: If no valid tiles_overview images were found for the dataset/bag.
    """
    ts = timestamp or datetime.now()
    collection = collect_tiles_overview_entries(dataset=dataset, bag_id=bag_id)
    if not collection.entries:
        raise RuntimeError(
            f"No tiles_overview entries found for dataset='{dataset.name}' and bag_id='{bag_id}'."
        )

    out_path = Path(output_path) if output_path is not None else _default_output_path(dataset, bag_id, ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries_sorted = sorted(collection.entries, key=lambda e: e.slide_id)

    # Front page shows previews of the first 2 entries only.
    front_page_preview_count = min(2, len(entries_sorted))
    remaining_entries = entries_sorted[front_page_preview_count:]

    total_pages = 1 + (int(ceil(len(remaining_entries) / 4.0)) if remaining_entries else 0)

    c = canvas.Canvas(str(out_path), pagesize=page_size, pageCompression=1)
    page_w, page_h = page_size

    _draw_front_page(
        c=c,
        page_w=page_w,
        page_h=page_h,
        dataset=dataset,
        bag_id=bag_id,
        generated_at=ts,
        collection=collection,
        total_pages=total_pages,
    )

    for chunk_index in range(0, len(remaining_entries), 4):
        page_no = 2 + (chunk_index // 4)
        page_entries = remaining_entries[chunk_index : chunk_index + 4]
        _draw_overview_page(
            c=c,
            page_w=page_w,
            page_h=page_h,
            bag_id=bag_id,
            entries=page_entries,
            page_no=page_no,
            total_pages=total_pages,
        )

    c.save()

    logger.info(
        "[TilesReport] Wrote PDF: %s (dataset=%s, bag_id=%s, included=%d, missing_overview=%d, pages=%d)",
        out_path,
        dataset.name,
        bag_id,
        collection.stats.included_slides,
        collection.stats.missing_overview,
        total_pages,
    )
    return out_path


def collect_tiles_overview_entries(*, dataset: WSIDataset, bag_id: str) -> TilesReportCollection:
    """Collect reportable tiles_overview entries from dataset samples in dataset order."""
    entries: list[TilesOverviewEntry] = []

    missing_overview = 0
    missing_coords = 0
    unreadable_h5 = 0
    corrupt_overview = 0
    representative_tiling_spec: dict[str, Any] | None = None

    for wsi in dataset.samples:
        h5_path = wsi.artifact_path
        if not h5_path.exists():
            missing_overview += 1
            continue

        try:
            with FileHandleH5(h5_path, mode="r") as slide_artifact:
                if not tiles_io.tiles_overview_exists(slide_artifact, bag_id):
                    missing_overview += 1
                    continue

                try:
                    image_bytes = bytes(tiles_io.read_tiles_overview(slide_artifact, bag_id))
                except Exception:
                    corrupt_overview += 1
                    logger.warning(
                        "[TilesReport] Failed to read tiles_overview for slide=%s bag_id=%s (%s)",
                        wsi.slide,
                        bag_id,
                        h5_path,
                        exc_info=True,
                    )
                    continue

                # Validate image bytes early so bad entries do not break PDF rendering.
                if not _is_valid_image_bytes(image_bytes):
                    corrupt_overview += 1
                    logger.warning(
                        "[TilesReport] tiles_overview is not a readable image for slide=%s bag_id=%s (%s)",
                        wsi.slide,
                        bag_id,
                        h5_path,
                    )
                    continue

                num_tiles: int | None = None
                try:
                    if tiles_io.coords_exist(slide_artifact, bag_id):
                        num_tiles = int(tiles_io.coords_num_rows(slide_artifact, bag_id))
                    else:
                        missing_coords += 1
                except Exception:
                    missing_coords += 1
                    logger.warning(
                        "[TilesReport] Failed to read coords count for slide=%s bag_id=%s (%s)",
                        wsi.slide,
                        bag_id,
                        h5_path,
                        exc_info=True,
                    )

                tiling_spec: dict[str, Any] | None = None
                try:
                    if tiles_io.tiling_spec_exists(slide_artifact, bag_id):
                        tiling_spec = tiles_io.read_tiling_spec(slide_artifact, bag_id)
                        if representative_tiling_spec is None and isinstance(tiling_spec, dict):
                            representative_tiling_spec = dict(tiling_spec)
                except Exception:
                    logger.warning(
                        "[TilesReport] Failed to read tiling_spec for slide=%s bag_id=%s (%s)",
                        wsi.slide,
                        bag_id,
                        h5_path,
                        exc_info=True,
                    )

                entries.append(
                    TilesOverviewEntry(
                        slide_id=wsi.slide,
                        artifact_path=h5_path,
                        image_bytes=image_bytes,
                        num_tiles=num_tiles,
                        tiling_spec=tiling_spec,
                    )
                )
        except Exception:
            unreadable_h5 += 1
            logger.warning(
                "[TilesReport] Failed to open/read H5 for slide=%s bag_id=%s (%s)",
                wsi.slide,
                bag_id,
                h5_path,
                exc_info=True,
            )

    stats = TilesReportStats(
        total_slides_expected=len(dataset.samples),
        included_slides=len(entries),
        missing_overview=missing_overview,
        missing_coords=missing_coords,
        unreadable_h5=unreadable_h5,
        corrupt_overview=corrupt_overview,
    )
    return TilesReportCollection(
        entries=entries,
        stats=stats,
        bag_id=bag_id,
        representative_tiling_spec=representative_tiling_spec,
    )


def _default_output_path(dataset: WSIDataset, bag_id: str, ts: datetime) -> Path:
    safe_bag_id = bag_id.replace("/", "_")
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    return Path(dataset.artifacts_dir) / f"tiles_report__{safe_bag_id}__{stamp}.pdf"


def _draw_front_page(
    *,
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    dataset: WSIDataset,
    bag_id: str,
    generated_at: datetime,
    collection: TilesReportCollection,
    total_pages: int,
) -> None:
    margin_x = 28
    top_y = page_h - 22

    # ---- small top banner ----
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_w / 2, top_y, "Intended for Research Use Only")

    # ---- title row ----
    title_y = top_y - 36
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin_x, title_y, "Tile extraction report")
    c.setFont("Helvetica", 18)
    c.drawRightString(page_w - margin_x, title_y, "PathBench")

    # ---- divider ----
    line_y = title_y - 10
    c.setLineWidth(0.8)
    c.setStrokeColor(colors.black)
    c.line(margin_x, line_y, page_w - margin_x, line_y)

    # ---- generated timestamp ----
    meta_y = line_y - 18
    c.setFont("Helvetica", 10)
    c.drawString(margin_x, meta_y, f"Generated: {generated_at.strftime('%m/%d/%Y %H:%M:%S')}")

    # ---- top boxes row (summary + histogram) ----
    table_x = margin_x
    box_y_top = meta_y - 44
    total_w = page_w - 2 * margin_x
    gap = 18
    table_w = (total_w - gap) * 0.48
    hist_w = total_w - gap - table_w
    hist_x = table_x + table_w + gap
    box_h = 118

    # titles above each box
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(table_x + table_w / 2, box_y_top + 10, "Run metadata")
    c.drawCentredString(hist_x + hist_w / 2, box_y_top + 10, "Tile count distribution")

    summary_rows = _build_front_page_summary_rows(dataset=dataset, bag_id=bag_id, collection=collection)
    _draw_key_value_table(c, x=table_x, y_top=box_y_top, w=table_w, h=box_h, rows=summary_rows)

    _draw_tile_count_histogram(
        c,
        x=hist_x,
        y_top=box_y_top,
        w=hist_w,
        h=box_h,
        entries=collection.entries,
    )

    # ---- preview images (2-up, slideflow-like first page preview) ----
    preview_y_top = box_y_top - box_h - 22
    previews = sorted(collection.entries, key=lambda e: e.slide_id)[:2]

    if previews:
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(page_w / 2, preview_y_top + 10, "Example slide overviews")

        gap = 18
        card_w = (page_w - 2 * margin_x - gap) / 2
        card_h = 230
        for i, entry in enumerate(previews):
            card_x = margin_x + i * (card_w + gap)
            _draw_overview_card(c, x=card_x, y_top=preview_y_top, w=card_w, h=card_h, entry=entry, draw_border=False)

    _draw_page_footer(c, page_w=page_w, y=18, page_no=1, total_pages=total_pages)
    c.showPage()


def _draw_overview_page(
    *,
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    bag_id: str,
    entries: list[TilesOverviewEntry],
    page_no: int,
    total_pages: int,
) -> None:
    margin_x = 28
    margin_top = 24
    margin_bottom = 28
    header_h = 14
    footer_h = 18
    grid_gap_x = 18
    grid_gap_y = 16

    # Minimal header for later pages.
    top_y = page_h - margin_top
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, top_y, "Tile extraction report")
    c.setFont("Helvetica", 10)
    c.drawRightString(page_w - margin_x, top_y, f"bag: {bag_id}")
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(margin_x, top_y - 4, page_w - margin_x, top_y - 4)

    grid_top = top_y - header_h - 4
    usable_h = grid_top - (margin_bottom + footer_h)
    usable_w = page_w - 2 * margin_x

    cell_w = (usable_w - grid_gap_x) / 2
    cell_h = (usable_h - grid_gap_y) / 2

    positions = [
        (margin_x, grid_top),
        (margin_x + cell_w + grid_gap_x, grid_top),
        (margin_x, grid_top - cell_h - grid_gap_y),
        (margin_x + cell_w + grid_gap_x, grid_top - cell_h - grid_gap_y),
    ]

    for entry, (x, y_top) in zip(entries, positions):
        _draw_overview_card(c, x=x, y_top=y_top, w=cell_w, h=cell_h, entry=entry, draw_border=False)

    _draw_page_footer(c, page_w=page_w, y=18, page_no=page_no, total_pages=total_pages)
    c.showPage()


def _build_front_page_summary_rows(
    *, dataset: WSIDataset, bag_id: str, collection: TilesReportCollection
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    spec = collection.representative_tiling_spec or {}
    rows.append(("bag_id", bag_id))

    # Compact summary first.
    preferred_keys = ["tile_px", "tile_mpp", "stride_px", "coord_space", "backend"]
    for key in preferred_keys:
        if key in spec:
            rows.append((key, _fmt_value(spec[key])))

    rows.extend(
        [
            ("total_slides", str(collection.stats.total_slides_expected)),
            ("slides_included", str(collection.stats.included_slides)),
            ("missing_overview", str(collection.stats.missing_overview)),
            ("missing_coords", str(collection.stats.missing_coords)),
            ("unreadable_h5", str(collection.stats.unreadable_h5)),
            ("corrupt_overview", str(collection.stats.corrupt_overview)),
        ]
    )

    # Then include the rest of the tiling spec (flattened).
    existing_keys = {k for k, _ in rows}
    for k, v in _flatten_dict(spec):
        if k in existing_keys:
            continue
        rows.append((k, _fmt_value(v)))

    return rows


def _draw_key_value_table(
    c: canvas.Canvas,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    rows: list[tuple[str, str]],
) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.rect(x, y_top - h, w, h, stroke=1, fill=0)

    # inner layout
    pad_x = 8
    pad_y = 8
    col_gap = 14
    line_h = 13

    col_w = (w - 2 * pad_x - col_gap) / 2
    col1_x = x + pad_x
    col2_x = col1_x + col_w + col_gap
    text_top_y = y_top - pad_y - 1

    rows_per_col = max(1, int((h - 2 * pad_y) // line_h))
    max_rows = rows_per_col * 2
    display_rows = rows[:max_rows]

    # vertical divider between columns (subtle structure)
    divider_x = col1_x + col_w + (col_gap / 2)
    c.setLineWidth(0.4)
    c.line(divider_x, y_top - h + 4, divider_x, y_top - 4)

    def _draw_cell(tx: float, ty: float, key: str, value: str) -> None:
        key_txt = f"{key}:"
        c.setFont("Helvetica-Bold", 8.4)
        c.drawString(tx, ty, key_txt)
        key_w = c.stringWidth(key_txt, "Helvetica-Bold", 8.4)

        c.setFont("Helvetica", 8.4)
        val_txt = _truncate_to_width(
            c,
            str(value),
            max_width=col_w - key_w - 6,
            font_name="Helvetica",
            font_size=8.4,
        )
        c.drawString(tx + key_w + 3, ty, val_txt)

    for idx, (k, v) in enumerate(display_rows):
        col = 0 if idx < rows_per_col else 1
        row_idx = idx if col == 0 else idx - rows_per_col
        tx = col1_x if col == 0 else col2_x
        ty = text_top_y - row_idx * line_h
        _draw_cell(tx, ty, k, v)

    if len(rows) > max_rows:
        c.setFont("Helvetica-Oblique", 7.5)
        c.drawRightString(x + w - 6, y_top - h + 5, f"+{len(rows) - max_rows} more")


def _draw_tile_count_histogram(
    c: canvas.Canvas,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    entries: Iterable[TilesOverviewEntry],
    n_bins: int = 16,
) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.rect(x, y_top - h, w, h, stroke=1, fill=0)

    tile_counts = [int(e.num_tiles) for e in entries if e.num_tiles is not None]

    if not tile_counts:
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(x + w / 2, y_top - h / 2, "No tile counts available")
        return

    left_pad = 28
    right_pad = 10
    top_pad = 18
    bottom_pad = 22
    px0 = x + left_pad
    py0 = y_top - h + bottom_pad
    pw = max(10.0, w - left_pad - right_pad)
    ph = max(10.0, h - top_pad - bottom_pad)

    counts_sorted = sorted(tile_counts)
    min_v = counts_sorted[0]
    max_v = counts_sorted[-1]

    if min_v == max_v:
        bins = [len(tile_counts)]
    else:
        step = (max_v - min_v) / float(n_bins)
        if step <= 0:
            step = 1.0
        bins = [0 for _ in range(n_bins)]
        for v in tile_counts:
            if v == max_v:
                idx = n_bins - 1
            else:
                idx = int((v - min_v) / step)
                idx = max(0, min(n_bins - 1, idx))
            bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_gap = 1.0
    bar_w = max(1.0, (pw - bar_gap * (len(bins) - 1)) / max(1, len(bins)))

    # Axes
    c.setLineWidth(0.5)
    c.line(px0, py0, px0, py0 + ph)
    c.line(px0, py0, px0 + pw, py0)

    # Y labels
    c.setFont("Helvetica", 6.8)
    c.drawRightString(px0 - 4, py0 - 2, "0")
    c.drawRightString(px0 - 4, py0 + ph - 2, str(max_count))

    c.saveState()
    c.translate(x + 9, py0 + ph / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Number of slides")
    c.restoreState()

    # Bars (gray/blue toned)
    c.setFillColor(colors.HexColor("#5f87a8"))
    c.setStrokeColor(colors.HexColor("#4d6f8c"))
    for i, count in enumerate(bins):
        bx = px0 + i * (bar_w + bar_gap)
        bh = 0 if max_count <= 0 else (count / max_count) * ph
        c.rect(bx, py0, bar_w, bh, stroke=1, fill=1)

    # X labels
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.setFont("Helvetica", 6.8)
    c.drawString(px0, py0 - 12, str(min_v))
    c.drawRightString(px0 + pw, py0 - 12, str(max_v))
    c.drawCentredString(px0 + pw / 2, py0 - 12, "Tile count per slide")


def _draw_overview_card(
    c,
    *,
    x: float,
    y_top: float,
    w: float,
    h: float,
    entry,
    draw_border: bool = True,
) -> None:
    """
    Draw one overview card:
    - centered slide_id + tile count (PDF text, fixed size)
    - overview image below (loaded from H5-stored JPEG bytes)

    Uses direct JPEG-bytes -> ImageReader first to keep PDFs small.
    Falls back to PIL decode only if needed.
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    if draw_border:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(x, y_top - h, w, h, stroke=1, fill=0)

    # ---- layout ----
    pad_x = 6
    pad_bottom = 4
    title_h = 26  # reserved area for 2 lines of PDF text

    # ---- text ----
    slide_id = str(getattr(entry, "slide_id", "unknown"))
    num_tiles = int(getattr(entry, "num_tiles", 0) or 0)

    max_text_w = w - 2 * pad_x
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 8.5)

    slide_text = slide_id
    while slide_text and c.stringWidth(slide_text, "Helvetica-Bold", 8.5) > max_text_w:
        if len(slide_text) <= 4:
            break
        slide_text = slide_text[:-4] + "..."
    if not slide_text:
        slide_text = "..."

    c.drawCentredString(x + w / 2, y_top - 2, slide_text)

    c.setFont("Helvetica-Bold", 8.0)
    c.drawCentredString(x + w / 2, y_top - 12, f"{num_tiles} tiles")

    # ---- image box ----
    img_x = x + pad_x
    img_y_top = y_top - title_h
    img_w = max(8.0, w - 2 * pad_x)
    img_h = max(8.0, h - title_h - pad_bottom)

    img_payload = (
        getattr(entry, "overview_bytes", None)
        or getattr(entry, "tiles_overview_bytes", None)
        or getattr(entry, "image_bytes", None)
    )

    if img_payload is None:
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(x + w / 2, y_top - h / 2, "No overview image")
        return

    # ---- normalize to Python bytes ----
    try:
        if isinstance(img_payload, np.ndarray):
            if img_payload.dtype != np.uint8:
                img_payload = img_payload.astype(np.uint8, copy=False)
            img_bytes = img_payload.tobytes()
        elif isinstance(img_payload, (bytes, bytearray, memoryview)):
            img_bytes = bytes(img_payload)
        else:
            img_bytes = bytes(img_payload)
    except Exception:
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(x + w / 2, y_top - h / 2, "Unreadable overview image")
        return

    # ---- try direct JPEG embedding first (keeps PDF small) ----
    img_reader = None
    iw = ih = None

    try:
        img_reader = ImageReader(BytesIO(img_bytes))
        iw, ih = img_reader.getSize()
        if iw <= 0 or ih <= 0:
            raise ValueError("Invalid image size from JPEG bytes")
    except Exception:
        # Fallback path: decode with PIL (more robust, but may increase PDF size)
        try:
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                raise ValueError("Invalid PIL image size")
            img_reader = ImageReader(pil_img)
        except Exception:
            c.setFont("Helvetica-Oblique", 8)
            c.drawCentredString(x + w / 2, y_top - h / 2, "Unreadable overview image")
            return

    # ---- draw fitted ----
    scale = min(img_w / float(iw), img_h / float(ih))
    draw_w = iw * scale
    draw_h = ih * scale

    draw_x = img_x + (img_w - draw_w) / 2.0
    draw_y = (img_y_top - img_h) + (img_h - draw_h) / 2.0

    c.drawImage(
        img_reader,
        draw_x,
        draw_y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )


def _fit_rect_preserve_aspect(
    *, img_w: float, img_h: float, box_x: float, box_y: float, box_w: float, box_h: float
) -> tuple[float, float, float, float]:
    if img_w <= 0 or img_h <= 0:
        return box_x, box_y, box_w, box_h

    scale = min(box_w / img_w, box_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    draw_x = box_x + (box_w - draw_w) / 2
    draw_y = box_y + (box_h - draw_h) / 2
    return draw_x, draw_y, draw_w, draw_h


def _draw_page_footer(c: canvas.Canvas, *, page_w: float, y: float, page_no: int, total_pages: int) -> None:
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawCentredString(page_w / 2, y, f"Page {page_no} of {total_pages}")


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key in sorted(d.keys()):
        value = d[key]
        out_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, out_key))
        else:
            items.append((out_key, value))
    return items


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def _truncate_to_width(
    c: canvas.Canvas,
    text: str,
    *,
    max_width: float,
    font_name: str,
    font_size: float,
) -> str:
    if max_width <= 0:
        return ""
    if c.stringWidth(text, font_name, font_size) <= max_width:
        return text
    ell = "..."
    ell_w = c.stringWidth(ell, font_name, font_size)
    if ell_w > max_width:
        return ""
    out = text
    while out and c.stringWidth(out, font_name, font_size) + ell_w > max_width:
        out = out[:-1]
    return out + ell


def _is_valid_image_bytes(data: bytes) -> bool:
    try:
        img = ImageReader(io.BytesIO(data))
        w, h = img.getSize()
        return bool(w and h)
    except Exception:
        return False