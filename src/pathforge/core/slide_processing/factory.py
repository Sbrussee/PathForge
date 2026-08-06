"""Construction helpers for registered slide-processing backends."""

from __future__ import annotations

from importlib import import_module

from pathforge.core.slide_processing.base import SlideProcessorBase


def build_slide_processor(backend_name: str) -> SlideProcessorBase:
    """Load and construct the registered processor for one backend name.

    Args:
        backend_name: ``slide_processing.backend`` registry key, such as
            ``"lazyslide"``.

    Returns:
        A newly constructed processor for the selected backend.

    Raises:
        ValueError: If the backend module cannot be imported or does not
            register a processor under ``backend_name``.

    Example:
        >>> processor = build_slide_processor("lazyslide")
    """
    from pathforge.utils.registries import SLIDE_PROCESSORS

    if not SLIDE_PROCESSORS.is_available(backend_name):
        try:
            import_module(f"pathforge.core.slide_processing.{backend_name}")
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Slide processing backend '{backend_name}' is not available."
            ) from exc

    if not SLIDE_PROCESSORS.is_available(backend_name):
        raise ValueError(f"Slide processing backend '{backend_name}' is not registered.")

    processor = SLIDE_PROCESSORS.get(backend_name)()
    if not isinstance(processor, SlideProcessorBase):
        raise TypeError(
            f"Slide processing backend '{backend_name}' did not construct a "
            "SlideProcessorBase instance."
        )
    return processor
