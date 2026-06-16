"""Backward-compatibility shim.

The implementation moved into the ``anima_mixer`` package. Import node
classes and mappings from there; this module re-exports the public surface
for older code that did ``from <plugin>.nodes import ...``.
"""

from .anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .anima_mixer.nodes_core import (
    AnimaArtistBasic,
    AnimaArtistCrossAttn,
    AnimaArtistPack,
    AnimaArtistProbe,
    AnimaArtistProbeReport,
)
from .anima_mixer.nodes_ui import (
    AnimaArtistChainBuilder,
    AnimaArtistChainPreview,
    AnimaArtistInspector,
    AnimaArtistOptions,
    AnimaArtistPreset,
    AnimaArtistRecipeLoad,
    AnimaArtistRecipeSave,
    AnimaArtistStarter,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "AnimaArtistBasic",
    "AnimaArtistCrossAttn",
    "AnimaArtistPack",
    "AnimaArtistProbe",
    "AnimaArtistProbeReport",
    "AnimaArtistChainBuilder",
    "AnimaArtistChainPreview",
    "AnimaArtistInspector",
    "AnimaArtistOptions",
    "AnimaArtistPreset",
    "AnimaArtistRecipeLoad",
    "AnimaArtistRecipeSave",
    "AnimaArtistStarter",
]
