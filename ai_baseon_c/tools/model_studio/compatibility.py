"""Restricted automatic compatibility for this project's historical CNN files.

This is deliberately a GUI policy layered over the converter's opt-in loader.
It neither imports a training script nor falls back to unrestricted pickle.
"""

import torch

from model_converter.loaders import load_checkpoint
from model_converter.schema import ConversionError


_LEGACY_IDENTITY = "__main__.CNNVoiceClassifier"
_LEGACY_GLOBALS = frozenset({
    _LEGACY_IDENTITY,
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv1d",
    "torch.nn.modules.batchnorm.BatchNorm1d",
    "torch.nn.modules.activation.ReLU",
    "torch.nn.modules.pooling.MaxPool1d",
    "torch.nn.modules.flatten.Flatten",
    "torch.nn.modules.dropout.Dropout",
    "torch.nn.modules.linear.Linear",
    "builtins.set",
})


def load_studio_checkpoint(path, *, trusted_module=False, auto_legacy=True):
    """Return ``(state, metadata, sha256, notices)`` without arbitrary execution.

    Tensor checkpoints always use the original weights-only path first. Only a
    statically identified historical CNN may retry the existing allowlisted
    weights-only loader, whose exact module types and graph are then checked.
    ``trusted_module`` remains available to existing callers; it never permits
    an unrestricted loader. ``auto_legacy=False`` restores strict opt-in policy.
    """
    if trusted_module:
        state, metadata, digest = load_checkpoint(path, trusted_module=True)
        return state, metadata, digest, []
    try:
        state, metadata, digest = load_checkpoint(path, trusted_module=False)
        return state, metadata, digest, []
    except ConversionError as initial_error:
        if not auto_legacy:
            raise
        try:
            globals_found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        except Exception:
            # Missing/corrupt/non-ZIP files preserve their original actionable
            # error. Static inspection is not a reason to permit extra globals.
            raise initial_error from None
        if _LEGACY_IDENTITY not in globals_found or not globals_found <= _LEGACY_GLOBALS:
            raise initial_error from None
    # Keep this outside the initial exception handler: an invalid historical
    # graph should report its graph failure, not the first missing-global error.
    state, metadata, digest = load_checkpoint(path, trusted_module=True)
    return state, metadata, digest, [
        "已自动兼容本工程旧版 CNN 整模型：使用已知类白名单和受限加载，"
        "并检查模型结构；原始文件未修改。"
    ]
