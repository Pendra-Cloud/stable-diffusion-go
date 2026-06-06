# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this repo is

`stable-diffusion-go` is a **pure-Go binding** to
[`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp),
built on [`github.com/ebitengine/purego`](https://github.com/ebitengine/purego)
so that **no cgo is required**. It calls the C++ shared library at runtime via
FFI and exposes a Go API for text-to-image, image-to-image, and text-to-video
generation across Linux, macOS, and Windows.

The module path is `github.com/Pendra-Cloud/stable-diffusion-go`.

## Layout

- `stable_diffusion.go` — root package: the high-level, ergonomic wrapper
  (`StableDiffusion`, `NewStableDiffusion`, `GenerateImage`, `GenerateVideo`,
  `Free`). Re-exports `Load`.
- `pkg/sd/` — the low-level binding:
  - `load.go` — `Load(libDir)` plus `registerFunctions`, which binds every
    native symbol via purego.
  - `load_library_unix.go` / `load_library_windows.go` — platform `openLibrary`
    / `closeLibrary` (build-tagged).
  - `cfree_unix.go` / `cfree_windows.go` — best-effort libc `free(3)` binding
    (build-tagged) used to release native-allocated buffers.
  - `free.go` — `FreeImage` / `FreeImages` helpers.
  - `stable_diffusion.go` — context, params structs, and method bindings.
  - `utils.go` — image conversion (`EncodePNG`, `SaveImage`, `toRGBA`), I/O, GPU
    detection.
- `examples/txt2img/`, `examples/txt2vid/` — one `package main` per directory.
- `lib/` — version pin and license text only. The actual `.so`/`.dylib`/`.dll`
  are **not** committed; they come from
  `leejet/stable-diffusion.cpp` releases matching `lib/version.txt`.

## Conventions to preserve

- **Lazy, caller-controlled loading.** Importing a package must do **no**
  filesystem access and **no** `dlopen`. The shared library is loaded only when
  the caller invokes `Load(libDir)` (empty `libDir` falls back to the OS search
  path). `Load` is idempotent and concurrency-safe.
- **Never panic across the FFI boundary.** A missing/incompatible library, a
  missing symbol, or malformed input must return an `error`, not crash. Convert
  any `purego` panic into an error (see the `recover` in `Load`).
- **Free native memory.** Buffers returned by the C library (e.g. the image
  array from `generate_image`) are not GC-managed — free them via
  `FreeImage`/`FreeImages`. The libc `free` binding is best-effort and degrades
  to a safe no-op when unavailable, so callers must tolerate that.
- **Reuse loaded contexts.** A context (`NewStableDiffusion` / `sd.NewContext`)
  loads a multi-GB model; reuse it across generations and free it once via
  `Free`. Do not free per request.
- **Symbol completeness.** `registerFunctions` binds the full symbol set; the
  native library must export all of them or `Load` fails. Keep the binding and
  `lib/version.txt` in lockstep with the upstream commit they target.
- **Cross-platform build tags.** Changes must keep `GOOS=linux`, `darwin`, and
  `windows` all compiling. Use the existing `_unix.go` / `_windows.go` split for
  platform-specific code rather than `runtime.GOOS` branching where a build tag
  is clearer.

## Testing (please add coverage with your changes)

**Add or update tests alongside any code change**, and keep existing tests
green. Match the existing style: table/round-trip tests living next to the code
(`pkg/sd/*_test.go`).

- Tests must **not require the native shared library** — it isn't present in CI
  or this repo. Cover logic that runs without it (param mapping, image
  conversion, error/no-op paths), and skip or guard anything that needs a loaded
  lib (see `load_test.go`, `free_test.go`).
- Prefer asserting that exported APIs **return errors** for malformed input
  rather than panicking.

Before pushing, run:

```bash
gofmt -l .            # must print nothing
go vet ./...
go build ./...        # builds the library and examples/...
go test ./...
# and confirm the build tags still compile on other platforms:
GOOS=windows GOARCH=amd64 go build ./...
GOOS=darwin  GOARCH=arm64 go build ./...
```

## Notes

- This is a **public** repository — keep everything here generic to the library.
  Do not add deployment specifics, internal infrastructure details, private
  roadmap, or issue-tracker references.
- License is MIT.
