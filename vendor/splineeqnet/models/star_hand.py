"""STAR-Hand model entry point.

STAR-Hand is the time-domain, spatio-temporally anisotropic CARD variant.
The implementation lives in :mod:`models.card`; this module gives the
architecture its own stable model entry point without duplicating the shared
training-compatible implementation.
"""

from models.card import CardConfig, CardForecaster

STARHandConfig = CardConfig
STARHandForecaster = CardForecaster

__all__ = [
    "CardConfig",
    "CardForecaster",
    "STARHandConfig",
    "STARHandForecaster",
]
