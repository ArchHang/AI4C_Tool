"""Emit typed, reentrant, caller-owned C workspaces and immutable model parameters."""

import json
from pathlib import Path
import re

import numpy as np

from .quantization import quantize_rows
from .schema import ConversionError, validate_portable_storage


def float_literal(value):
    text = format(float(np.float32(value)), ".9g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def c_array(name, array):
    array = np.asarray(array)
    if array.dtype == np.float32:
        ctype, values = "float", [float_literal(v) for v in array.reshape(-1)]
    else:
        ctype = "int8_t" if array.dtype == np.int8 else "int32_t"
        values = [str(int(v)) for v in array.reshape(-1)]
    lines = [f"static const {ctype} {name}[{array.size}] = {{"]
    for start in range(0, len(values), 8):
        lines.append("    " + ", ".join(values[start:start + 8]) + ",")
    return "\n".join(lines + ["};\n"])


def quantize_parameters(spec, weight_scaling="maxabs", mse_min_ratio=0.5, mse_steps=101):
    quantized, metrics = {}, {}
    for name, array in spec.arrays.items():
        if name.endswith("weight") or name in ("w_ih", "w_hh"):
            q, scale, rowsum, error = quantize_rows(array, weight_scaling, mse_min_ratio, mse_steps)
            quantized[name + "_q"] = q
            quantized[name + "_scale"] = scale
            if name in ("w_ih", "w_hh"):
                quantized[name + "_rowsum"] = rowsum
            metrics[name] = error
    return quantized, metrics


def weight_cache_plan(spec, precision, cache_policy="none"):
    """Select small Conv kernels; larger or recurrent layers retain Flash weights."""
    if cache_policy not in ("none", "packed"):
        raise ConversionError("cache_policy must be none or packed")
    if cache_policy == "none" or precision == "fp32" or spec.architecture != "cnn":
        return []
    layers = []
    for index, shape in enumerate(spec.dimensions["conv_shapes"]):
        outputs, inputs = shape[:2]
        kernel = inputs * 5
        if 1 <= kernel <= 64 and 1 <= outputs <= 32:
            elements = ((kernel + 3) & ~3) * outputs
            layers.append({"layer": index, "kernel_elements": kernel,
                           "output_channels": outputs, "elements": elements,
                           "bytes": elements * 2})
    return layers


def weight_cache_metadata(spec, precision, cache_policy="none"):
    layers = weight_cache_plan(spec, precision, cache_policy)
    return {"policy": cache_policy, "bytes": sum(layer["bytes"] for layer in layers),
            "state_bytes": 4 if layers else 0, "layers": layers,
            "accelerated_backend": "arm_dsp", "other_backends": "native fallback"}


def workspace_members(spec, precision, cache_policy="none"):
    d = spec.dimensions
    members = []
    if precision in ("int8", "both"):
        members.append(("uint8_t", "input_q", d["input_elements"]))
        if spec.architecture == "cnn":
            pooled = [s[0] * s[4] for s in d["conv_shapes"]]
            members.extend((("uint8_t", "q_a", max(pooled[::2])),
                            ("uint8_t", "q_b", pooled[1]),
                            ("int32_t", "accumulators", max(pooled)),
                            ("float", "fc_hidden", d["hidden_size"]),
                            ("uint8_t", "fc_hidden_q", d["hidden_size"])))
        else:
            h, t = d["hidden_size"], d["seq_length"]
            members.extend((("uint8_t", "hidden_q", h * t),
                            ("float", "hidden_scales", t), ("int", "hidden_zero_points", t),
                            ("float", "int8_work", h if spec.architecture == "rnn" else h * 6)))
    if precision in ("fp32", "both"):
        if spec.architecture == "cnn":
            sizes = [s[0] * s[3] for s in d["conv_shapes"]]
            members.extend((("float", "f_a", max(sizes[::2])),
                            ("float", "f_b", sizes[1]),
                            ("float", "float_hidden", d["hidden_size"])))
        else:
            members.append(("float", "hidden", d["seq_length"] * d["hidden_size"]))
            if spec.architecture == "lstm":
                members.append(("float", "cell", d["hidden_size"]))
    cache = weight_cache_plan(spec, precision, cache_policy)
    if cache:
        # The aligned word and arrays of four int16s keep cache starts four-byte aligned.
        members.append(("int32_t", "weight_cache_ready", 1))
        members.extend(("int16_t", f"conv{layer['layer']}_packed", layer["elements"])
                       for layer in cache)
    return members


def emit_header(spec, prefix, precision, members):
    upper = prefix.upper()
    declarations = []
    def entry_doc(description):
        return (f"/**\n * @brief {description}\n"
                " * @param input [in] Complete contiguous input, disjoint from workspace/logits.\n"
                f" * @param input_elements [in] Exactly {upper}_INPUT_ELEMENTS.\n"
                f" * @param logits [out] At least {upper}_OUTPUT_ELEMENTS floats.\n"
                " * @param logits_capacity [in] Actual logits capacity in float elements.\n"
                " * @param workspace [in,out] Complete aligned storage; separate for each caller.\n"
                " * @return CNN_NORMAL or a CNN pointer/data error. States reset for each call.\n */\n")
    if precision in ("fp32", "both"):
        declarations.append(entry_doc("Run the FP32 graph; finite input and output are required.") +
                            f"int32_t {prefix}_run_fp32(const float *input, size_t input_elements,\n"
                            f"                          float *logits, size_t logits_capacity,\n"
                            f"                          {prefix}_workspace_t *workspace);")
    if precision in ("int8", "both"):
        declarations.extend((entry_doc("Run prequantized uint8 input using the manifest scale/zero point.") +
                             f"int32_t {prefix}_run_int8(const uint8_t *input, size_t input_elements,\n"
                             f"                          float *logits, size_t logits_capacity,\n"
                             f"                          {prefix}_workspace_t *workspace);",
                             entry_doc("Quantize finite float input, then run the int8 graph.") +
                             f"int32_t {prefix}_run_quantized(const float *input, size_t input_elements,\n"
                             f"                               float *logits, size_t logits_capacity,\n"
                             f"                               {prefix}_workspace_t *workspace);"))
    fields = "\n".join(f"    {ctype} {name}[{count}];" for ctype, name, count in members)
    declarations_text = "\n".join(declarations)
    return f'''/**
 * @file model.h
 * @brief Generated {spec.schema} model. Parameters are immutable; each caller owns its workspace.
 */
#ifndef AI_GENERATED_{upper}_MODEL_H_
#define AI_GENERATED_{upper}_MODEL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define {upper}_INPUT_ELEMENTS {spec.dimensions["input_elements"]}u
#define {upper}_OUTPUT_ELEMENTS {spec.dimensions["num_classes"]}u

typedef struct {prefix}_workspace {{
{fields}
}} {prefix}_workspace_t;

/** @brief Query workspace bytes including padding. @return Bytes on this target ABI. */
size_t {prefix}_workspace_size(void);
/** @brief Query workspace alignment. @return Required alignment in bytes on this target ABI. */
size_t {prefix}_workspace_alignment(void);
/**
 * @brief Validate storage, clear states and initialize optional weight caches once.
 * @param workspace [in,out] Caller storage aligned to the queried requirement.
 * @param workspace_bytes [in] Actual capacity in bytes, at least the queried size.
 * @return CNN_NORMAL or a CNN pointer/data error.
 */
int32_t {prefix}_init({prefix}_workspace_t *workspace, size_t workspace_bytes);
/**
 * @brief Infer one batch-free input. Recurrent states start at zero for each call.
 * @param input [in] Contiguous channel-major CNN or [time][feature] recurrent input.
 * @param input_elements [in] Must equal {upper}_INPUT_ELEMENTS exactly.
 * @param logits [out] Output logits; disjoint from input and workspace.
 * @param logits_capacity [in] At least {upper}_OUTPUT_ELEMENTS float elements.
 * @param workspace [in,out] Initialized, aligned caller-owned storage; no concurrent reuse.
 * @return CNN_NORMAL on success; CNN_POINTER_NULL/CNN_DATA_EXCEPTION on invalid input.
 * Float inputs and results must be finite. Separate workspaces support concurrent callers.
 */
int32_t {prefix}_run(const float *input, size_t input_elements,
                    float *logits, size_t logits_capacity, {prefix}_workspace_t *workspace);
{declarations_text}

#ifdef __cplusplus
}}
#endif
#endif
'''


def emit_common(prefix, cache=()):
    initialization = ""
    if cache:
        calls = []
        for layer in cache:
            index = layer["layer"]
            calls.append(f'''    {{
        const conv2d_filter_i8_t filter = {{1u, 5u, {layer['kernel_elements'] // 5}u,
            {layer['output_channels']}u, {prefix}_conv{index}_weight_q,
            {prefix}_conv{index}_weight_scale, {prefix}_conv{index}_bias}};
        const int32_t status = conv2d_int8_pack_weights(&filter,
            workspace->conv{index}_packed, sizeof(workspace->conv{index}_packed));
        if (status != CNN_NORMAL) {{
            return status;
        }}
    }}''')
        initialization = "\n".join(calls) + "\n    workspace->weight_cache_ready[0] = 1;\n"
    return f'''
typedef struct {prefix}_aligned_probe {{
    uint8_t byte;
    {prefix}_workspace_t workspace;
}} {prefix}_aligned_probe_t;
typedef struct {prefix}_float_probe {{
    uint8_t byte;
    float value;
}} {prefix}_float_probe_t;

size_t {prefix}_workspace_size(void)
{{
    return sizeof({prefix}_workspace_t);
}}

size_t {prefix}_workspace_alignment(void)
{{
    return offsetof({prefix}_aligned_probe_t, workspace);
}}

int32_t {prefix}_init({prefix}_workspace_t *workspace, size_t workspace_bytes)
{{
    if (workspace == NULL) {{
        return CNN_POINTER_NULL;
    }}
    if (workspace_bytes < sizeof(*workspace)
        || (uintptr_t)workspace % {prefix}_workspace_alignment() != 0u) {{
        return CNN_DATA_EXCEPTION;
    }}
    memset(workspace, 0, sizeof(*workspace));
{initialization}\
    return CNN_NORMAL;
}}

static int32_t {prefix}_overlaps(const void *first, size_t first_bytes,
                               const void *second, size_t second_bytes)
{{
    const uintptr_t first_address = (uintptr_t)first;
    const uintptr_t second_address = (uintptr_t)second;
    if (first_address > UINTPTR_MAX - first_bytes || second_address > UINTPTR_MAX - second_bytes) {{
        return 1;
    }}
    return first_address < second_address + second_bytes
        && second_address < first_address + first_bytes;
}}

static int32_t {prefix}_check(const void *input, size_t input_bytes,
                            float *logits, size_t capacity, {prefix}_workspace_t *workspace)
{{
    if (input == NULL || logits == NULL || workspace == NULL) {{
        return CNN_POINTER_NULL;
    }}
    if (capacity < {prefix.upper()}_OUTPUT_ELEMENTS
        || (uintptr_t)logits % offsetof({prefix}_float_probe_t, value) != 0u
        || (uintptr_t)workspace % {prefix}_workspace_alignment() != 0u
        || {prefix}_overlaps(input, input_bytes, workspace, sizeof(*workspace))
        || {prefix}_overlaps(logits, {prefix.upper()}_OUTPUT_ELEMENTS * sizeof(float),
                            workspace, sizeof(*workspace))
        || {prefix}_overlaps(input, input_bytes, logits,
                            {prefix.upper()}_OUTPUT_ELEMENTS * sizeof(float))) {{
        return CNN_DATA_EXCEPTION;
    }}
    return CNN_NORMAL;
}}

static int32_t {prefix}_finite(const float *values, size_t elements)
{{
    if ((uintptr_t)values % offsetof({prefix}_float_probe_t, value) != 0u) {{
        return CNN_DATA_EXCEPTION;
    }}
    for (size_t index = 0; index < elements; index++) {{
        if (!isfinite(values[index])) {{
            return CNN_DATA_EXCEPTION;
        }}
    }}
    return CNN_NORMAL;
}}
'''


def emit_cnn(spec, prefix, precision, scale, zero_point, cache=()):
    d, code = spec.dimensions, []
    p = prefix
    if precision in ("int8", "both"):
        for i, shape in enumerate(d["conv_shapes"]):
            out_ch, in_ch = shape[:2]
            code.append(f"static const conv2d_filter_i8_t {p}_conv{i} = "
                        f"{{1u, 5u, {in_ch}u, {out_ch}u, {p}_conv{i}_weight_q, "
                        f"{p}_conv{i}_weight_scale, {p}_conv{i}_bias}};")
        code.append(f'''
int32_t {p}_run_int8(const uint8_t *input, size_t input_elements,
                    float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    int32_t status;
    quant_param_t activation = {{{float_literal(scale)}, {zero_point}}};
    conv2d_data_i8_t feature = {{1u, {d['input_length']}u, {d['input_channels']}u,
                                (uint8_t *)input}};
    conv2d_data_i8_t output = {{0u, 0u, 0u, workspace != NULL ? workspace->q_a : NULL}};
    conv2d_config_i8_t config = {{1u, 0u, &{p}_conv0, activation, 1, &activation}};
    if (input_elements != {p.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    status = {p}_check(input, input_elements, logits, logits_capacity, workspace);
    if (status != CNN_NORMAL) {{
        return status;
    }}
''')
        if cache:
            code.append('''    if (workspace->weight_cache_ready[0] != 1) {
        return CNN_DATA_EXCEPTION;
    }''')
        cached_layers = {layer["layer"] for layer in cache}
        for i in range(3):
            target = "q_a" if i % 2 == 0 else "q_b"
            if i in cached_layers:
                call = f'''    {{
        const conv2d_packed_context_t packed = {{workspace->conv{i}_packed,
            sizeof(workspace->conv{i}_packed), workspace->accumulators,
            sizeof(workspace->accumulators)}};
        status = conv2d_int8_maxpool2_run_packed(&feature, &config, &output, &packed);
    }}'''
            else:
                call = '''    status = conv2d_int8_maxpool2_run_with_scratch(&feature, &config, &output,
                                                 workspace->accumulators,
                                                 sizeof(workspace->accumulators));'''
            code.append(f'''    config.filter = &{p}_conv{i};
    config.in_act = activation;
    output.data = workspace->{target};
{call}
    if (status != CNN_NORMAL) {{
        return status;
    }}
    feature = output;''')
        code.append(f'''    status = linear_int8(feature.data, {d['flatten_size']}u, {p}_fc1_weight_q,
                         {p}_fc1_weight_scale, {p}_fc1_bias, {d['hidden_size']}u,
                         activation.scale, activation.zp, workspace->fc_hidden);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    float minimum = workspace->fc_hidden[0];
    float maximum = minimum;
    for (size_t index = 0; index < {d['hidden_size']}u; index++) {{
        const float value = workspace->fc_hidden[index];
        if (!isfinite(value)) {{
            return CNN_DATA_EXCEPTION;
        }}
        if (value < minimum) {{
            minimum = value;
        }}
        if (value > maximum) {{
            maximum = value;
        }}
    }}
    activation.scale = fmaxf((maximum - minimum) / 255.0f, 1e-9f);
    if (!isfinite(activation.scale)) {{
        return CNN_DATA_EXCEPTION;
    }}
    if (maximum - minimum < 255.0f * 1e-9f) {{
        /* Preserve a nonzero constant rather than collapsing it into the epsilon floor. */
        const float magnitude = fmaxf(fabsf(minimum), fabsf(maximum));
        activation.scale = magnitude > 1e-9f ? magnitude / 128.0f : 1e-9f;
        activation.zp = minimum < 0.0f ? 255 : 0;
    }} else {{
        /* A uint8 affine range must include real zero. Clamping zp alone would
         * clip same-sign vectors, e.g. [1,2] into the representable range [0,1]. */
        minimum = fminf(minimum, 0.0f);
        maximum = fmaxf(maximum, 0.0f);
        activation.scale = fmaxf((maximum - minimum) / 255.0f, 1e-9f);
        const float zero_point = fminf(fmaxf(-minimum / activation.scale, 0.0f), 255.0f);
        activation.zp = (int)lrintf(zero_point);
    }}
    quantize_u8(workspace->fc_hidden, {d['hidden_size']}u, activation.scale, activation.zp,
                workspace->fc_hidden_q);
    status = linear_int8(workspace->fc_hidden_q, {d['hidden_size']}u, {p}_fc2_weight_q,
                         {p}_fc2_weight_scale, {p}_fc2_bias, {d['num_classes']}u,
                         activation.scale, activation.zp, logits);
    return status == CNN_NORMAL ? {p}_finite(logits, {p.upper()}_OUTPUT_ELEMENTS) : status;
}}
''')
    if precision in ("fp32", "both"):
        # Folded parameters are evaluated through the existing vectorizable float convolution.
        # gamma=sqrt(BN_EPS), variance=0 make its normalization factor exactly one.
        for i, shape in enumerate(d["conv_shapes"]):
            out_ch, in_ch = shape[:2]
            code.append(c_array(f"{p}_conv{i}_zeros", np.zeros(out_ch, np.float32)))
            code.append(c_array(f"{p}_conv{i}_identity", np.full(out_ch,
                                         np.sqrt(np.float32(1e-5)), np.float32)))
            code.append(f'''static const conv2d_filter_t {p}_float_conv{i} =
    {{1u, 5u, {in_ch}u, {out_ch}u, (float *){p}_conv{i}_weight}};
static const batch_norm2d_t {p}_float_bn{i} =
    {{{out_ch}u, (float *){p}_conv{i}_zeros, (float *){p}_conv{i}_zeros,
      (float *){p}_conv{i}_identity, (float *){p}_conv{i}_bias}};''')
        code.append(f'''
int32_t {p}_run_fp32(const float *input, size_t input_elements,
                    float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    int32_t status;
    conv2d_data_t feature = {{1u, {d['input_length']}u, {d['input_channels']}u, (float *)input}};
    conv2d_data_t output = {{0u, 0u, 0u, NULL}};
    max_pool_config_t pool = {{1u, 2u, 2u, 0u}};
    conv2d_config_t config = {{1u, 0u, NULL, NULL}};
    if (input_elements != {p.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    status = {p}_check(input, input_elements * sizeof(float), logits, logits_capacity, workspace);
    if (status != CNN_NORMAL || {p}_finite(input, input_elements) != CNN_NORMAL) {{
        return status != CNN_NORMAL ? status : CNN_DATA_EXCEPTION;
    }}
''')
        for i in range(3):
            code.append(f'''    config.filter = (conv2d_filter_t *)&{p}_float_conv{i};
    config.bn = (batch_norm2d_t *)&{p}_float_bn{i};
    output.data = workspace->{'f_a' if i % 2 == 0 else 'f_b'};
    status = conv2d_bn_no_bias(&feature, &config, &output);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    status = leaky_relu(0.0f, output.data, output.channel * output.col, output.data);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    feature = output;
    status = maxpool(&feature, &pool, &output);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    feature = output;''')
        code.append(f'''    linear_param_t head1 = {{{d['flatten_size']}u, {d['hidden_size']}u,
                             (float *){p}_fc1_weight, (float *){p}_fc1_bias}};
    linear_param_t head2 = {{{d['hidden_size']}u, {d['num_classes']}u,
                             (float *){p}_fc2_weight, (float *){p}_fc2_bias}};
    status = linear_layer(feature.data, &head1, workspace->float_hidden);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    status = linear_layer(workspace->float_hidden, &head2, logits);
    return status == CNN_NORMAL ? {p}_finite(logits, {p.upper()}_OUTPUT_ELEMENTS) : status;
}}
''')
    return "\n".join(code)


def emit_recurrent(spec, prefix, precision, scale, zero_point):
    d, p, kind = spec.dimensions, prefix, spec.architecture
    i, h, c, t = (d[key] for key in ("input_size", "hidden_size", "num_classes", "seq_length"))
    code = []
    if precision in ("fp32", "both"):
        biases = f"{p}_b_ih, {p}_w_hh, {p}_b_hh" if kind == "rnn" else f"{p}_w_hh, {p}_bias"
        code.append(f'''static const {kind}_fp32_param_t {p}_float_parameters =
    {{{i}u, {h}u, {p}_w_ih, {biases}}};
static const {kind}_out_fp32_t {p}_float_head =
    {{{h}u, {c}u, {p}_out_weight, {p}_out_bias}};

int32_t {p}_run_fp32(const float *input, size_t input_elements,
                    float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    int32_t status;
    if (input_elements != {p.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    status = {p}_check(input, input_elements * sizeof(float), logits, logits_capacity, workspace);
    if (status != CNN_NORMAL || {p}_finite(input, input_elements) != CNN_NORMAL) {{
        return status != CNN_NORMAL ? status : CNN_DATA_EXCEPTION;
    }}
    status = {kind}_forward_fp32(&{p}_float_parameters, &{p}_float_head, input, {t}u,
                                 {"workspace->cell, " if kind == "lstm" else ""}workspace->hidden, logits);
    return status == CNN_NORMAL ? {p}_finite(logits, {p.upper()}_OUTPUT_ELEMENTS) : status;
}}
''')
    if precision in ("int8", "both"):
        biases = (f"{p}_b_ih, {p}_w_hh_q, {p}_w_hh_scale, {p}_w_hh_rowsum, {p}_b_hh"
                  if kind == "rnn" else
                  f"{p}_w_hh_q, {p}_w_hh_scale, {p}_w_hh_rowsum, {p}_bias")
        code.append(f'''static const {kind}_int8_param_t {p}_int8_parameters =
    {{{i}u, {h}u, {p}_w_ih_q, {p}_w_ih_scale, {p}_w_ih_rowsum, {biases}}};
static const {kind}_out_int8_t {p}_int8_head =
    {{{h}u, {c}u, {p}_out_weight_q, {p}_out_weight_scale, {p}_out_bias}};

int32_t {p}_run_int8(const uint8_t *input, size_t input_elements,
                    float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    int32_t status;
    if (input_elements != {p.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    status = {p}_check(input, input_elements, logits, logits_capacity, workspace);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    status = {kind}_forward_int8(&{p}_int8_parameters, &{p}_int8_head, input, {t}u,
                                 {float_literal(scale)}, {zero_point},
                                 {"workspace->int8_work, " if kind == "lstm" else ""}workspace->hidden_q,
                                 workspace->hidden_scales, workspace->hidden_zero_points,
                                 {"workspace->int8_work, " if kind == "rnn" else ""}logits);
    return status == CNN_NORMAL ? {p}_finite(logits, {p.upper()}_OUTPUT_ELEMENTS) : status;
}}
''')
    return "\n".join(code)


def emit_entry_points(prefix, precision, scale, zero_point):
    p = prefix
    code = []
    if precision in ("int8", "both"):
        code.append(f'''
int32_t {p}_run_quantized(const float *input, size_t input_elements,
                         float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    int32_t status;
    if (input_elements != {p.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    status = {p}_check(input, input_elements * sizeof(float), logits, logits_capacity, workspace);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    if ((uintptr_t)input % offsetof({p}_float_probe_t, value) != 0u) {{
        return CNN_DATA_EXCEPTION;
    }}
    /* Validate and quantize in one pass. Divide in float before rounding,
     * exactly as quantize_u8; reciprocal multiplication can change ties. */
    for (size_t index = 0; index < input_elements; index++) {{
        const float value = input[index];
        if (!isfinite(value)) {{
            return CNN_DATA_EXCEPTION;
        }}
        const float normalized = value / {float_literal(scale)};
        if (normalized <= {float_literal(-zero_point)}) {{
            workspace->input_q[index] = 0u;
        }} else if (normalized >= {float_literal(255 - zero_point)}) {{
            workspace->input_q[index] = 255u;
        }} else {{
            workspace->input_q[index] = (uint8_t)((int32_t)lrintf(normalized) + {zero_point});
        }}
    }}
    /* input_q is owned by this wrapper, so copy-free internal dispatch bypasses its alias check. */
    uint8_t *quantized_input = workspace->input_q;
''')
        # The public int8 entry rejects any overlap. Its input_q member is deliberate internal overlap:
        # use a dedicated internal inference helper generated by a later replacement in emit_model.
        code.append(f'''    return {p}_infer_int8(quantized_input, input_elements,
                            logits, logits_capacity, workspace);
}}
''')
    delegate = "run_quantized" if precision == "int8" else "run_fp32"
    code.append(f'''
int32_t {p}_run(const float *input, size_t input_elements,
               float *logits, size_t logits_capacity, {p}_workspace_t *workspace)
{{
    return {p}_{delegate}(input, input_elements, logits, logits_capacity, workspace);
}}
''')
    return "\n".join(code)


def emit_model(spec, output_dir, prefix="ai_model", precision="int8", input_scale=1 / 255,
               input_zero_point=0, checkpoint_sha256=None, weight_scaling="maxabs",
               mse_min_ratio=0.5, mse_steps=101, cache_policy="none"):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ConversionError("prefix must be a lowercase C identifier beginning with a letter")
    if precision not in ("int8", "fp32", "both"):
        raise ConversionError("precision must be int8, fp32 or both")
    validate_portable_storage(spec)
    scale = np.float32(input_scale)
    if (not np.isfinite(scale) or scale <= 0 or not isinstance(input_zero_point, int)
            or isinstance(input_zero_point, bool) or not 0 <= input_zero_point <= 255):
        raise ConversionError("Input quantization requires finite positive float32 scale and zp in [0,255]")
    cache = weight_cache_plan(spec, precision, cache_policy)
    members = workspace_members(spec, precision, cache_policy)
    if any(count <= 0 or count > 0x1fffffff for _, _, count in members):
        raise ConversionError("Generated buffer dimensions exceed portable 32-bit address-space bounds")
    workspace_lower_bound = sum(count * {"uint8_t": 1, "int16_t": 2}.get(ctype, 4)
                                for ctype, _, count in members)
    if workspace_lower_bound > 0x7fffff00:
        raise ConversionError("Combined workspace exceeds portable signed 32-bit size bounds")
    try:
        quantized, metrics = quantize_parameters(spec, weight_scaling, mse_min_ratio, mse_steps)
    except ValueError as error:
        raise ConversionError(str(error)) from error
    arrays = {}
    for name, value in spec.arrays.items():
        if precision in ("fp32", "both") or not (name.endswith("weight") or name in ("w_ih", "w_hh")):
            arrays[name] = value
    if precision in ("int8", "both"):
        arrays.update(quantized)
    source = [f'''/** @file model.c @brief Generated {spec.schema} parameters and graph. */
#include "model.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ai_baseon_c/cnn.h"
''' + (f'#include "ai_baseon_c/{spec.architecture}.h"\n' if spec.architecture != "cnn" else "")]
    source.extend(c_array(prefix + "_" + name, value) for name, value in arrays.items())
    source.append(emit_common(prefix, cache))
    graph = (emit_cnn(spec, prefix, precision, scale, input_zero_point, cache) if spec.architecture == "cnn"
             else emit_recurrent(spec, prefix, precision, scale, input_zero_point))
    if precision in ("int8", "both"):
        # Expose a strict checked public API and use an internal graph for float->q dispatch.
        graph = graph.replace(f"int32_t {prefix}_run_int8(", f"static int32_t {prefix}_infer_int8(", 1)
        start = graph.index(f"    status = {prefix}_check(input, input_elements, logits, logits_capacity, workspace);")
        end = graph.index("    }", start) + len("    }")
        graph = graph[:start] + "    (void)logits_capacity;" + graph[end:]
        public = f'''
int32_t {prefix}_run_int8(const uint8_t *input, size_t input_elements,
                         float *logits, size_t logits_capacity, {prefix}_workspace_t *workspace)
{{
    if (input_elements != {prefix.upper()}_INPUT_ELEMENTS) {{
        return CNN_DATA_EXCEPTION;
    }}
    const int32_t status = {prefix}_check(input, input_elements, logits, logits_capacity, workspace);
    if (status != CNN_NORMAL) {{
        return status;
    }}
    return {prefix}_infer_int8(input, input_elements, logits, logits_capacity, workspace);
}}
'''
        graph += public
    source.extend((graph, emit_entry_points(prefix, precision, scale, input_zero_point)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.h").write_text(emit_header(spec, prefix, precision, members), encoding="utf-8")
    (output / "model.c").write_text("\n".join(source), encoding="utf-8")
    example = f'''/** @file example.c @brief Minimal hosted example; replace input with application features. */
#include "model.h"

#include <stdio.h>

static {prefix}_workspace_t workspace;
static const float input[{prefix.upper()}_INPUT_ELEMENTS] = {{0.0f}};

int main(void)
{{
    float logits[{prefix.upper()}_OUTPUT_ELEMENTS];
    int32_t status = {prefix}_init(&workspace, sizeof(workspace));
    if (status == 0) {{
        status = {prefix}_run(input, {prefix.upper()}_INPUT_ELEMENTS,
                              logits, {prefix.upper()}_OUTPUT_ELEMENTS, &workspace);
    }}
    if (status != 0) {{
        fprintf(stderr, "inference error %ld\\n", (long)status);
        return 1;
    }}
    for (size_t index = 0; index < {prefix.upper()}_OUTPUT_ELEMENTS; index++) {{
        printf("%.9g%c", (double)logits[index],
               index + 1u == {prefix.upper()}_OUTPUT_ELEMENTS ? '\\n' : ' ');
    }}
    return 0;
}}
'''
    (output / "example.c").write_text(example, encoding="utf-8")
    parameter_bytes = sum(value.nbytes for value in arrays.values())
    manifest = {"format_version": 1, "architecture": spec.architecture, "schema": spec.schema,
                "prefix": prefix, "precision": precision, "checkpoint_sha256": checkpoint_sha256,
                "dimensions": spec.dimensions, "input_quantization": {"scale": float(scale),
                "zero_point": input_zero_point}, "parameter_bytes": parameter_bytes,
                "workspace_members": [{"type": ctype, "name": name, "elements": count}
                                      for ctype, name, count in members],
                "workspace_size": "query <prefix>_workspace_size() on target ABI",
                "workspace_alignment": "query <prefix>_workspace_alignment() on target ABI",
                "weight_cache": weight_cache_metadata(spec, precision, cache_policy),
                "weight_quantization_error": metrics,
                "weight_quantization": {"scale_selection": weight_scaling,
                    "mse_min_ratio": mse_min_ratio if weight_scaling == "mse" else None,
                    "mse_steps": mse_steps if weight_scaling == "mse" else None,
                    "rounding": "ties_to_even", "per_row_mse_not_above_maxabs": True},
                "runtime_allocation": "none", "input_layout": "CHW, batch=1" if spec.architecture == "cnn"
                else "[time][feature], batch=1, initial hidden/cell zero",
                "activation_quantization": "CNN ReLU max/255 dynamic; FC1 affine includes zero "
                "with ties-even zp and magnitude/128 constant fallback; "
                "recurrent hidden affine per time step with runtime's truncating zp",
                "bn_eps": spec.metadata.get("bn_eps", [1e-5] * 3) if spec.architecture == "cnn" else None}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    readme = f'''Generated graph: {spec.schema}, precision: {precision}.

Build with the portable library (path below is relative to this directory; adapt it):
  cc -std=c99 -O3 -ffp-contract=off -fno-fast-math -I/path/to/ai_baseon_c/include -I. \\
     model.c example.c /path/to/ai_baseon_c/src/cnn.c \\
     {('/path/to/ai_baseon_c/src/' + spec.architecture + '.c ') if spec.architecture != 'cnn' else ''}/path/to/ai_baseon_c/src/platform.c -lm -o model_example

Or link libai_baseon_c.a. Automatic backend selection follows compiler target macros.
Use static {prefix}_workspace_t storage (or separately aligned caller storage), initialize it,
then call {prefix}_run with exact input elements and output capacity. Separate workspaces
are reentrant. No runtime heap, OS, UART or preprocessing is required.
Weight cache policy: {cache_policy}; expanded Conv weights use {sum(layer['bytes'] for layer in cache)}
bytes plus {4 if cache else 0} bytes of initialization state (workspace query includes ABI padding).
Call init once before inference, and again after modifying workspace storage. The packed cache
accelerates supported Cortex-M DSP Conv layers; other platforms keep their native backend.
The original immutable int8 weights remain in Flash. Fully connected weights are not cached.
int8 inputs represent x = (q - {input_zero_point}) * {float(scale):.9g}.
Floating-point quantized input saturates values outside that range. Use --input-scale and
--input-zero-point for a different preprocessing contract. This exporter does not calculate FFT.
For recurrent models, *_run* resets state for each fixed-length sequence; hidden outputs,
per-time-step scales and zero points remain in the public workspace for verification.
--precision both includes both paths and both sets of buffers/weights for validation;
use int8 for a compact deployment. Preserve ordinary IEEE float semantics; do not use fast-math.
Source schema validation cannot infer arbitrary Python control flow from tensor names.
See ai_baseon_c/docs/model_conversion.md for schemas, safe loading and accuracy verification.
'''
    (output / "README.txt").write_text(readme, encoding="utf-8")
    return manifest
