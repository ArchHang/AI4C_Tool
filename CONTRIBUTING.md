# Contributing

## Scope

AI4C Tool maintains a C inference library and a Python/Tk model-conversion
application. Read the [user guide](docs/guide.md),
[project layout](docs/project_layout.md), and [license notices](NOTICE.md)
before proposing a change. Source publication does not imply a new license grant
for unannotated components.

## Development

1. Create a branch from `main` and keep the change focused.
2. Use Python 3.10 or newer and the documented NumPy/PyTorch dependencies.
3. Run `python ai_baseon_c/tools/check_repository.py --build-system cmake`.
   Use `--require-gui` when a display and Tk are available.
4. For an operator, backend, quantization rule, or schema change, add an
   independent numerical reference or a regression that distinguishes the bug.
   Include relevant size, overflow, alignment, repeated-call, and boundary cases.
5. Explain the change, validation, compatibility impact, and any unverified
   platforms in the pull request.

Keep C code in `ai_baseon_c/src` and public headers in
`ai_baseon_c/include/ai_baseon_c`. Compatibility forwarding files are not a second
implementation. Preserve C11 portability, caller-owned workspaces, and the
documented floating-point and rounding contracts. Do not enable fast-math to
obtain a speed result that changes numerical semantics.

Changes to supported model graphs must update the schema parser, emitter,
analysis, compression dependencies, and conversion tests together. An unknown
state dictionary must not be presented as a supported inference graph.

## Evidence and documentation

Keep the README short. Put usage and numerical definitions in the user guide;
put experiments, methods, results, and limitations in the benchmark report.
Use repository-relative links and documented commands. Report parameter count,
weight bytes, total firmware size, and runtime memory separately.

Performance comparisons must use the same checkpoint, input, clock, compiler,
timing boundary, and runtime configuration for the factor being compared.
Distinguish host execution, target compilation, instruction emulation, and
physical measurements. Do not call an aggregate speaker accuracy a general
speaker-verification result.

## Files and cleanup

Keep generated exports, build products, and temporary files under `build/`,
`artifacts/`, or a task-owned `.work/` directory. Clean only files whose purpose
has ended; preserve inputs, final artifacts, key logs, and recovery evidence.
Follow [AGENTS.md](AGENTS.md) for Windows path and cleanup rules.

Before committing, review `git diff --cached --stat` and
`git diff --cached --check`. Do not commit recordings, real speaker checkpoints,
credentials, machine snapshots, generated executables, or SDK installations.
Add any public fixture by an explicit path and document its synthetic origin.
