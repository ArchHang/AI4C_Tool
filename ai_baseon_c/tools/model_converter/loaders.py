"""Load tensors safely; never import user training scripts to unpickle a model."""

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch import nn

from .schema import ConversionError


class CNNVoiceClassifier(nn.Module):
    """Compatibility identity for this repository's historical full-module checkpoints."""

    def forward(self, values):
        values = self.model(values)
        return self.fc2(self.dropout(self.fc1(values.reshape(values.shape[0], -1))))


# Historical train.py was executed as a script and saved this exact global name.
CNNVoiceClassifier.__module__ = "__main__"


def validate_cnn_module(module):
    if type(module) is not CNNVoiceClassifier:
        raise ConversionError("Full-module export supports only the known CNNVoiceClassifier class; "
                              "save other models as state_dict plus explicit schema metadata")
    expected = (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d,
                nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d,
                nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d, nn.Flatten)
    if type(module.model) is not nn.Sequential or len(module.model) != len(expected):
        raise ConversionError("CNN module does not match the supported 13-stage graph")
    for index, (layer, layer_type) in enumerate(zip(module.model, expected)):
        if type(layer) is not layer_type:
            raise ConversionError(f"Unsupported CNN stage {index}: {type(layer).__name__}")
        if layer_type is nn.Conv1d:
            if (layer.kernel_size != (5,) or layer.stride != (1,) or layer.padding != (0,)
                    or layer.dilation != (1,) or layer.groups != 1
                    or layer.padding_mode != "zeros"):
                raise ConversionError("Only valid conv5 stride1 dilation1 groups1 is supported")
        elif layer_type is nn.BatchNorm1d:
            if not layer.track_running_stats or not layer.affine:
                raise ConversionError("BatchNorm requires affine parameters and running statistics")
        elif layer_type is nn.MaxPool1d:
            if (layer.kernel_size != 2 or layer.stride != 2 or layer.padding != 0
                    or layer.dilation != 1 or layer.ceil_mode or layer.return_indices):
                raise ConversionError("Only floor maxpool2 stride2 is supported")
        elif layer_type is nn.Flatten:
            if layer.start_dim != 1 or layer.end_dim != -1:
                raise ConversionError("CNN Flatten must flatten all non-batch dimensions")
    if type(module.fc1) is not nn.Linear or type(module.fc2) is not nn.Linear:
        raise ConversionError("CNN classifier requires two Linear layers")
    if type(module.dropout) is not nn.Dropout:
        raise ConversionError("CNN dropout must be nn.Dropout (removed only in eval mode)")
    if set(module._modules) != {"model", "dropout", "fc1", "fc2"}:
        raise ConversionError("CNN contains unsupported extra modules")
    return {"schema": "cnn_voice_v1", "input_length": 2048,
            "bn_eps": [module.model[index].eps for index in (1, 5, 9)]}


def load_checkpoint(path, trusted_module=False):
    path = Path(path)
    if not path.is_file():
        raise ConversionError(f"Checkpoint not found: {path}")
    allowed = [CNNVoiceClassifier, nn.Sequential, nn.Conv1d, nn.BatchNorm1d,
               nn.ReLU, nn.MaxPool1d, nn.Flatten, nn.Dropout, nn.Linear, set]
    try:
        if trusted_module:
            # Still use the restricted weights-only unpickler, even after explicit opt-in.
            with torch.serialization.safe_globals(allowed):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ConversionError("Safe checkpoint loading failed. Save a tensor state_dict/checkpoint; "
                              "for this repository's trusted legacy CNN full module, use "
                              "--trusted-module. Arbitrary pickle classes are never executed. "
                              f"Detail: {error}") from error
    metadata = {}
    if isinstance(checkpoint, nn.Module):
        if not trusted_module:
            raise ConversionError("Full modules require --trusted-module")
        metadata = validate_cnn_module(checkpoint)
        checkpoint.eval()
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint or "model_state_dict" in checkpoint:
            state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
            metadata = checkpoint.get("metadata", checkpoint.get("model_config", {}))
            if not isinstance(metadata, Mapping):
                raise ConversionError("checkpoint metadata/model_config must be a mapping")
            metadata = dict(metadata)
        else:
            state = checkpoint
        if not isinstance(state, Mapping) or not state:
            raise ConversionError("state_dict must be a nonempty mapping")
    else:
        raise ConversionError("Expected tensor state_dict, checkpoint mapping, or known full module")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ConversionError("state_dict must map string keys to tensors only")
        key = key.removeprefix("module.")
        if key in normalized:
            raise ConversionError(f"Duplicate normalized tensor key: {key}")
        normalized[key] = value
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return normalized, metadata, digest.hexdigest()
