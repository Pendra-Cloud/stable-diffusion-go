# stable-diffusion-go

[简体中文](README-ZH.md)

A pure Golang binding library for `stable-diffusion.cpp` based on `github.com/ebitengine/purego`, **no cgo dependency required**, supporting cross-platform operation.

## 🌟 Project Features

- **Pure Go Implementation**: Based on the purego library, calls C++ dynamic libraries without cgo
- **Cross-platform Support**: Supports Windows, Linux, macOS, and other mainstream operating systems
- **Complete Functionality**: Implements the main APIs of stable-diffusion.cpp, including text-to-image, image-to-image, video generation, etc.
- **Simple and Easy to Use**: Provides a concise Go language API for easy integration into existing projects
- **High Performance**: Supports performance optimization features like FlashAttention and model quantization
- **Includes Precompiled Libraries**: Provides precompiled dynamic libraries for Windows platform, ready to use out of the box

## 📁 Project Structure

```
stable-diffusion-go/
├── examples/           # Example programs directory
│   ├── txt2img.go      # Text-to-image generation example
│   └── txt2vid.go      # Text-to-video generation example
├── lib/        # Dynamic library directory
│   ├── darwin/ # macOS platform dynamic library
│   │   └── libstable-diffusion.dylib
│   ├── linux/  # Linux platform dynamic library
│   │   └── libstable-diffusion.so
│   ├── windows/ # Windows platform dynamic library
│   │   ├── avx/      # AVX instruction set version
│   │   │   └── stable-diffusion.dll
│   │   ├── avx2/     # AVX2 instruction set version
│   │   │   └── stable-diffusion.dll
│   │   ├── avx512/   # AVX512 instruction set version
│   │   │   └── stable-diffusion.dll
│   │   ├── cuda12/   # CUDA 12 version
│   │   │   └── stable-diffusion.dll
│   │   ├── noavx/    # No AVX instruction set version
│   │   │   └── stable-diffusion.dll
│   │   ├── rocm/     # ROCm version
│   │   │   └── stable-diffusion.dll
│   │   └── vulkan/   # Vulkan version
│   │       └── stable-diffusion.dll
│   ├── ggml.txt
│   ├── stable-diffusion.cpp.txt
│   └── version.txt
├── pkg/                # Go package directory
│   └── sd/             # Core binding library
│       ├── load_library_unix.go   # Unix platform dynamic library loading
│       ├── load_library_windows.go # Windows platform dynamic library loading
│       ├── stable_diffusion.go    # Core functionality implementation
│       └── utils.go               # Auxiliary utility functions
├── .gitignore          # Git ignore file configuration
├── README.md           # Project documentation
├── go.mod              # Go module file
├── go.sum              # Go dependency checksum file
└── stable_diffusion.go # Root directory entry file
```
Note: All dynamic link library files in the lib directory need to be downloaded from https://github.com/leejet/stable-diffusion.cpp/releases according to the version in lib/version.txt

## 🚀 Quick Start

### 1. Install Dependencies

```bash
go get github.com/Pendra-Cloud/stable-diffusion-go
```

### 2. Prepare Model Files

Model files need to be prepared before use, supporting multiple formats:
- Diffusion models: `.gguf` format (e.g., z_image_turbo-Q4_K_M.gguf)
- LLM models: `.gguf` format (e.g., Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
- VAE models: `.safetensors` format (e.g., diffusion_pytorch_model.safetensors)

### 3. Dynamic Library Description

The project includes precompiled dynamic libraries for multiple platforms, located in the `pkg/sd/lib/` directory:
- **Windows**: Multiple versions to suit different hardware
  - `avx/`: Supports AVX instruction set
  - `avx2/`: Supports AVX2 instruction set
  - `avx512/`: Supports AVX512 instruction set
  - `cuda12/`: Supports CUDA 12
  - `noavx/`: No AVX instruction set dependency
  - `rocm/`: Supports ROCm
  - `vulkan/`: Supports Vulkan
- **Linux**: `libstable-diffusion.so`
- **macOS**: `libstable-diffusion.dylib`

The caller controls where and when the shared library is loaded. Call `Load`
once before creating a context or generating:

```go
// Load from a caller-supplied directory (e.g. where the worker ships its libs).
if err := stablediffusion.Load("/usr/lib/pendra"); err != nil {
	// The native lib is absent or incompatible — handle/skip the backend.
	// Load returns an error and never panics.
}

// Or pass an empty dir to fall back to the OS default library search path.
err := stablediffusion.Load("")
```

`Load` is lazy and idempotent: importing the package performs no filesystem
access and no `dlopen`. On Windows, the GPU/CPU variant subdirectories
(`cuda12/`, `rocm/`, `vulkan/`, `avx2/`, …) are resolved within the supplied
directory.

### 4. Run Examples

#### Text-to-Image Generation

```bash
# Enter the examples directory
cd examples

# Run text-to-image example
go run txt2img.go
```

Example code:

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	fmt.Println("Stable Diffusion Go - Text to Image Example")
	fmt.Println("===============================================")

	// Create Stable Diffusion instance
	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "path/to/diffusion_model.gguf",
		LLMPath:            "path/to/llm_model.gguf",
		VAEPath:            "path/to/vae_model.safetensors",
		DiffusionFlashAttn: true,
		OffloadParamsToCPU: true,
	})

	if err != nil {
		fmt.Println("Failed to create instance:", err)
		return
	}
	defer sd.Free()

	// Generate image
	err = sd.GenerateImage(&stablediffusion.ImgGenParams{
		Prompt:      "一位穿着明朝服饰的美女行走在花园中",
		Width:       512,
		Height:      512,
		SampleSteps: 10,
		CfgScale:    1.0,
	}, "output.png")

	if err != nil {
		fmt.Println("Failed to generate image:", err)
		return
	}

	fmt.Println("Image generated successfully!")
}
```
![](output_demo.png)

#### Text-to-Video Generation

```bash
# Run text-to-video example
go run txt2vid.go
```

## 📚 Core Features

### 1. Context Management

- Create and destroy Stable Diffusion contexts
- Support multiple model path configurations
- Provide rich performance optimization parameters

### 2. Text-to-Image Generation (txt2img)

- Generate high-quality images from text descriptions
- Support Chinese and English prompts
- Adjustable image dimensions, sampling steps, CFG scale, and other parameters
- Support random seed generation

### 3. Text-to-Video Generation (txt2vid)

- Generate videos from text prompts
- Support custom frame count and resolution
- Support Easycache optimization
- Integrate FFmpeg for video encoding

## 📝 Usage Guide

### Basic Usage

1. **Create Instance**: Use `NewStableDiffusion` to create a Stable Diffusion instance
2. **Configure Parameters**: Set context parameters and generation parameters
3. **Generate Content**: Call `GenerateImage` or `GenerateVideo` to generate content
4. **Release Resources**: Use `defer sd.Free()` to release resources

### Context Parameters Description

| Parameter Name | Type | Description |
|----------------|------|-------------|
| DiffusionModelPath | string | Diffusion model file path |
| LLMPath | string | LLM model file path |
| VAEPath | string | VAE model file path |
| NThreads | int | Number of threads |
| DiffusionFlashAttn | bool | Whether to enable FlashAttention |
| OffloadParamsToCPU | bool | Whether to offload some parameters to CPU |
| WType | SDType | Model quantization type |

### Image Generation Parameters Description

| Parameter Name | Type | Description |
|----------------|------|-------------|
| Prompt | string | Prompt text |
| NegativePrompt | string | Negative prompt text |
| Width | int | Image width |
| Height | int | Image height |
| Seed | int | Random seed |
| SampleSteps | int | Number of sampling steps |
| CfgScale | float64 | CFG scale |
| Strength | float64 | Initial image strength (img2img only) |

## 🔧 Performance Optimization

### 1. Adjust Thread Count

Adjust the `NThreads` parameter according to the number of CPU cores:

```go
ctxParams := &stablediffusion.ContextParams{
    // Other parameters...
    NThreads: 8, // Adjust according to CPU core count
}
```

### 2. Use Quantized Models

Using quantized models can improve performance and reduce memory usage:

```go
ctxParams := &stablediffusion.ContextParams{
    // Other parameters...
    WType: stablediffusion.SDTypeQ4_K, // Use Q4_K quantized model
}
```

### 3. Adjust Sampling Steps

Reducing the number of sampling steps can improve generation speed but may reduce image quality:

```go
imgGenParams := &stablediffusion.ImgGenParams{
    // Other parameters...
    SampleSteps: 10, // Reduce sampling steps
}
```

### 4. Enable FlashAttention

Enabling FlashAttention can accelerate the diffusion process:

```go
ctxParams := &stablediffusion.ContextParams{
    // Other parameters...
    DiffusionFlashAttn: true,
}
```

## ⚠️ Notes

1. **Dynamic Library Path**: The caller supplies the library directory via `Load(libDir)`; an empty `libDir` falls back to the OS default library search path
2. **Model Compatibility**: Ensure using model formats compatible with stable-diffusion.cpp
3. **Dependencies**: Install dependencies like CUDA or Vulkan as needed
4. **Video Generation**: Requires FFmpeg for video encoding
5. **Memory Usage**: Large models may require more memory, it is recommended to use quantized models
6. **About AMD Graphics Cards (Windows Platform)**: If using AMD graphics cards (including AMD integrated graphics), you need to download the ROCm library and place it in the project root directory, download link: https://github.com/leejet/stable-diffusion.cpp/releases/download/master-453-4ff2c8c/sd-master-4ff2c8c-bin-win-rocm-x64.zip
7. **About Vulkan**: If using non-nvidia graphics cards (such as AMD or Intel graphics cards, including integrated graphics), you can install Vulkan to enable GPU acceleration

## 📦 Example Programs

### Text-to-Image Example

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	// Create instance
	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "models/z_image_turbo-Q4_K_M.gguf",
		LLMPath:            "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
		VAEPath:            "models/diffusion_pytorch_model.safetensors",
		DiffusionFlashAttn: true,
	})
	if err != nil {
		fmt.Println("Failed to create instance:", err)
		return
	}
	defer sd.Free()

	// Generate image
	err = sd.GenerateImage(&stablediffusion.ImgGenParams{
		Prompt:      "A cute Corgi dog running on the grass",
		Width:       512,
		Height:      512,
		SampleSteps: 15,
		CfgScale:    2.0,
	}, "output_corgi.png")

	if err != nil {
		fmt.Println("Failed to generate image:", err)
		return
	}

	fmt.Println("Image generated successfully!")
}
```

### Text-to-Video Example

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	// Create instance
	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "D:\\hf-mirror\\wan2.1\\wan2.1_t2v_1.3B_bf16.safetensors",
		T5XXLPath:          "D:\\hf-mirror\\wan2.1\\umt5-xxl-encoder-Q4_K_M.gguf",
		VAEPath:            "D:\\hf-mirror\\wan2.1\\wan_2.1_vae.safetensors",
		DiffusionFlashAttn: true,
		KeepClipOnCPU:      true,
		OffloadParamsToCPU: true,
		NThreads:           4,
		FlowShift:          3.0,
	})

	if err != nil {
		fmt.Println("Failed to create stable diffusion instance:", err)
		return
	}
	defer sd.Free()

	err = sd.GenerateVideo(&stablediffusion.VidGenParams{
		Prompt:      "一个在长满桃花树下拍照的美女",
		Width:       300,
		Height:      300,
		SampleSteps: 40,
		VideoFrames: 33,
		CfgScale:    6.0,
	}, "./output.mp4")

	if err != nil {
		fmt.Println("Failed to generate video:", err)
		return
	}

	fmt.Println("Video generated successfully!")
}
```

## 📄 License

MIT License

## 🤝 Contribution

Welcome to submit Issues and Pull Requests!

## 🔗 Related Projects

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): C++ implementation of Stable Diffusion model
- [purego](https://github.com/ebitengine/purego): Go language FFI library without cgo

## 📞 Support

If you encounter problems during use, please:
1. Check the example code
2. Check the dynamic library path and model files
3. Check project Issues
4. Submit a new Issue

---

Thank you for using stable-diffusion-go! If this project has helped you, please give us a Star ⭐️
