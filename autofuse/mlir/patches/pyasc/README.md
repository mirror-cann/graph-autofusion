# PyAsc patches

PyAsc submodule follows upstream `https://gitcode.com/cann/pyasc.git`.
Autofuse-specific changes are maintained as ordered patches in this directory.

Rules:

- Do not point `.gitmodules` to a personal fork.
- Treat the PyAsc submodule gitlink as the pinned upstream baseline.
- Keep each patch reviewable and related to one PyAsc change.
- Export only selected Autofuse-required commits or ranges; do not export the full upstream history.
- Refresh patches after syncing upstream.
- Apply patches before building `ENABLE_AUTOFUSE_MLIR=ON`.
