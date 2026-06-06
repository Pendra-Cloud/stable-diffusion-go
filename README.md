# stable-diffusion-go

A **pure-Go** binding to [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp),
built on [`ebitengine/purego`](https://github.com/ebitengine/purego) so it calls
the native library directly over FFI — **no cgo required**. It runs on Linux,
macOS, and Windows and supports text-to-image, image-to-image, and
text-to-video generation.

> This project is a fork of [`orangelang/stable-diffusion-go`](https://github.com/orangelang/stable-diffusion-go).
> See [Acknowledgements](#acknowledgements).

## Highlights

- **No cgo** — pure Go via `purego`; the native shared library is loaded at runtime.
- **Caller-controlled loading** — you decide where and when the library is loaded; importing the package touches nothing.
- **Cross-platform** — Linux, macOS, and Windows, with GPU (CUDA / ROCm / Vulkan) and CPU (AVX) variant selection.
- **Broad API coverage** — txt2img, img2img, txt2vid, upscaling, and model conversion.

## Installation

```bash
go get github.com/Pendra-Cloud/stable-diffusion-go
```

## The native library

This repository contains **only the Go binding** — the native shared library is
not committed. The library is built from the upstream commit pinned in
[`lib/version.txt`](lib/version.txt) (currently `master-453-4ff2c8c`).

### Prebuilt archives

Per-variant archives are built and published from this repository's
[Releases](https://github.com/Pendra-Cloud/stable-diffusion-go/releases) by the
[`build-libs.yml`](.github/workflows/build-libs.yml) workflow (upstream ships no
prebuilt release covering these variants). Download the one matching your host,
extract it into a directory, and pass that directory to `Load`:

```
stable-diffusion-libs-linux-amd64-cpu-<version>.tar.gz      # also -cuda, -vulkan
stable-diffusion-libs-linux-arm64-cpu-<version>.tar.gz
stable-diffusion-libs-darwin-arm64-metal-<version>.tar.gz
stable-diffusion-libs-windows-amd64-<version>.tar.gz        # carries the subdir tree
```

Each archive ships a **single self-contained library** (ggml is statically
linked, with hidden visibility so only the `stable-diffusion.cpp` symbols are
exported and the embedded `ggml` symbols stay local — they won't collide with
another in-process library that has its own `ggml`). The Windows archive
contains the GPU/CPU variant subdirectories the loader selects from
(`avx2/`, `avx512/`, `avx/`, `noavx/`, `vulkan/`, `cuda12/`); extract it whole
and point `Load` at its root. CUDA archives need a host CUDA runtime; Vulkan
archives need a Vulkan loader/ICD.

| Platform | Library file |
| --- | --- |
| Linux | `libstable-diffusion.so` |
| macOS | `libstable-diffusion.dylib` |
| Windows | `stable-diffusion.dll` (under a variant subdir) |

To build it yourself instead, compile `stable-diffusion.cpp` at the pinned
commit with `-DSD_BUILD_SHARED_LIBS=ON`. The binding registers the full
`stable-diffusion.cpp` symbol set ([`lib/expected-symbols.txt`](lib/expected-symbols.txt)),
so the library must export all of them — keep the library version in lockstep
with `lib/version.txt`.

## Loading the library

The caller controls loading via `Load(libDir)`. Importing the package performs
**no** filesystem access and **no** `dlopen`; nothing native happens until you
call `Load`, which is lazy, idempotent, and returns an error (never panics) when
the library is missing or incompatible.

```go
import stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"

// Load from a directory you control...
if err := stablediffusion.Load("/path/to/libs"); err != nil {
    // The native library is absent or incompatible — handle gracefully.
    log.Fatal(err)
}

// ...or pass an empty dir to use the OS default library search path.
if err := stablediffusion.Load(""); err != nil {
    log.Fatal(err)
}
```

On Windows, GPU/CPU variant subdirectories (`cuda12/`, `rocm/`, `vulkan/`,
`avx2/`, …) are resolved within the supplied directory.

## Quick start

```go
package main

import (
    "log"

    stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
    // Resolve the native library (empty dir => OS default search path).
    if err := stablediffusion.Load(""); err != nil {
        log.Fatal(err)
    }

    // Create an instance — this loads the model and is reused across calls.
    sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
        DiffusionModelPath: "models/diffusion_model.gguf",
        LLMPath:            "models/llm_model.gguf",
        VAEPath:            "models/vae.safetensors",
        DiffusionFlashAttn: true,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer sd.Free() // free the loaded model once, when you're done

    // Generate an image. The instance can serve many GenerateImage calls.
    if err := sd.GenerateImage(&stablediffusion.ImgGenParams{
        Prompt:      "A cute Corgi running on the grass",
        Width:       512,
        Height:      512,
        SampleSteps: 15,
        CfgScale:    2.0,
    }, "output.png"); err != nil {
        log.Fatal(err)
    }
}
```

### Running the examples

Each example is its own `main` package:

```bash
go run ./examples/txt2img   # text-to-image
go run ./examples/txt2vid   # text-to-video (requires FFmpeg for encoding)
```

## Usage notes

- **Reuse the instance.** `NewStableDiffusion` loads a multi-GB model. Keep the
  instance and call `GenerateImage` repeatedly; release it once with
  `defer sd.Free()`. Don't create a new instance per request.
- **In-memory PNG.** Need the encoded bytes instead of a file? The low-level
  package offers `sd.EncodePNG(*sd.SDImage) ([]byte, error)`.
- **Freeing native images.** If you work with raw results from the low-level
  `pkg/sd` API, free the native-allocated buffers with `sd.FreeImage` /
  `sd.FreeImages`. The high-level `GenerateImage` already does this for you.
- **Video** generation uses FFmpeg to encode frames — make sure `ffmpeg` is on
  your `PATH`.
- **Performance.** Tune `NThreads`, enable `DiffusionFlashAttn`, and use
  quantized weights (e.g. `WType: stablediffusion.SDTypeQ4_K`) to trade quality
  for speed and memory.

## Project structure

```
stable-diffusion-go/
├── stable_diffusion.go   # high-level wrapper (NewStableDiffusion, GenerateImage, ...)
├── pkg/sd/               # low-level binding (Load, contexts, EncodePNG, FreeImage, ...)
├── examples/
│   ├── txt2img/          # text-to-image example
│   └── txt2vid/          # text-to-video example
└── lib/                  # version pin + license text (no native binaries)
```

## Contributing

Issues and pull requests are welcome. Please keep tests passing and add coverage
alongside your changes — see [CLAUDE.md](CLAUDE.md) for the conventions this repo
follows (lazy caller-controlled loading, no panics across the FFI boundary,
freeing native memory, and keeping all platforms compiling). Before opening a PR:

```bash
gofmt -l .        # should print nothing
go vet ./...
go build ./...
go test ./...
```

## Acknowledgements

Huge thanks to the projects and people this binding builds on:

- **[orangelang/stable-diffusion-go](https://github.com/orangelang/stable-diffusion-go)** by **foxaos** — the original Go binding this project is forked from.
- **[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** by **leejet** — the C++ Stable Diffusion implementation this binds to.
- **[ebitengine/purego](https://github.com/ebitengine/purego)** — the cgo-free FFI library that makes this possible.

## License

[MIT](LICENSE)
