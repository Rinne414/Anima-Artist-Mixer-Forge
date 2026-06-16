"""Anima Artist Mixer - multi-artist style mixing for the Anima model.

Hooks Anima/MiniTrainDIT cross-attention layers to mix independently encoded
artist conditionings, sidestepping LLM text-encoder interference when several
artist tags share one prompt. See CHANGELOG.md for version history.
"""

from .nodes_core import (
    AnimaArtistBasic,
    AnimaArtistCrossAttn,
    AnimaArtistPack,
    AnimaArtistProbe,
    AnimaArtistProbeReport,
)
from .nodes_ui import (
    AnimaArtistChainBuilder,
    AnimaArtistChainPreview,
    AnimaArtistInspector,
    AnimaArtistOptions,
    AnimaArtistPreset,
    AnimaArtistRecipeLoad,
    AnimaArtistRecipeSave,
    AnimaArtistStarter,
)

NODE_CLASS_MAPPINGS = {
    "AnimaArtistBasic": AnimaArtistBasic,
    "AnimaArtistStarter": AnimaArtistStarter,
    "AnimaArtistChainBuilder": AnimaArtistChainBuilder,
    "AnimaArtistChainPreview": AnimaArtistChainPreview,
    "AnimaArtistPack": AnimaArtistPack,
    "AnimaArtistCrossAttn": AnimaArtistCrossAttn,
    "AnimaArtistOptions": AnimaArtistOptions,
    "AnimaArtistPreset": AnimaArtistPreset,
    "AnimaArtistInspector": AnimaArtistInspector,
    "AnimaArtistRecipeSave": AnimaArtistRecipeSave,
    "AnimaArtistRecipeLoad": AnimaArtistRecipeLoad,
    "AnimaArtistProbe": AnimaArtistProbe,
    "AnimaArtistProbeReport": AnimaArtistProbeReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaArtistBasic": "Anima Artist Basic (Recommended)",
    "AnimaArtistStarter": "Anima Artist Starter",
    "AnimaArtistChainBuilder": "Anima Artist Chain Builder",
    "AnimaArtistChainPreview": "Anima Artist Chain Preview",
    "AnimaArtistPack": "Anima Artist Pack (Split + Encode)",
    "AnimaArtistCrossAttn": "Anima Artist Cross-Attn (v2)",
    "AnimaArtistOptions": "Anima Artist Options (Advanced)",
    "AnimaArtistPreset": "Anima Artist Preset (One Knob)",
    "AnimaArtistInspector": "Anima Artist Inspector",
    "AnimaArtistRecipeSave": "Anima Artist Recipe (Save)",
    "AnimaArtistRecipeLoad": "Anima Artist Recipe (Load)",
    "AnimaArtistProbe": "Anima Artist Layer Probe",
    "AnimaArtistProbeReport": "Anima Artist Probe Report",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
