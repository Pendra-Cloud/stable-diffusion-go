# stable-diffusion-go

[English](README.md)

一个基于 `github.com/ebitengine/purego` 实现的 `stable-diffusion.cpp` 纯 Golang 绑定库，**无需依赖 cgo**，支持跨平台运行。

## 🌟 项目特点

- **纯 Go 实现**：基于 purego 库，无需 cgo 即可调用 C++ 动态库
- **跨平台支持**：支持 Windows、Linux、macOS 等主流操作系统
- **完整功能**：实现了 stable-diffusion.cpp 的主要 API，包括文本到图像、图像到图像、视频生成等
- **简单易用**：提供简洁的 Go 语言 API，便于集成到现有项目中
- **高性能**：支持 FlashAttention、模型量化等性能优化特性
- **包含预编译库**：提供 Windows 平台预编译动态库，开箱即用

## 📁 项目结构

```
stable-diffusion-go/
├── examples/           # 示例程序目录
│   ├── txt2img.go      # 文本到图像生成示例
│   └── txt2vid.go      # 文本到视频生成示例
├── lib/        # 动态库目录
│   ├── darwin/ # macOS 平台动态库
│   │   └── libstable-diffusion.dylib
│   ├── linux/  # Linux 平台动态库
│   │   └── libstable-diffusion.so
│   ├── windows/ # Windows 平台动态库
│   │   ├── avx/      # AVX 指令集版本
│   │   │   └── stable-diffusion.dll
│   │   ├── avx2/     # AVX2 指令集版本
│   │   │   └── stable-diffusion.dll
│   │   ├── avx512/   # AVX512 指令集版本
│   │   │   └── stable-diffusion.dll
│   │   ├── cuda12/   # CUDA 12 版本
│   │   │   └── stable-diffusion.dll
│   │   ├── noavx/    # 无 AVX 指令集版本
│   │   │   └── stable-diffusion.dll
│   │   ├── rocm/     # ROCm 版本
│   │   │   └── stable-diffusion.dll
│   │   └── vulkan/   # Vulkan 版本
│   │       └── stable-diffusion.dll
│   ├── ggml.txt
│   ├── stable-diffusion.cpp.txt
│   └── version.txt
├── pkg/                # Go 包目录
│   └── sd/             # 核心绑定库
│       ├── load_library_unix.go   # Unix 平台动态库加载
│       ├── load_library_windows.go # Windows 平台动态库加载
│       ├── stable_diffusion.go    # 核心功能实现
│       └── utils.go               # 辅助工具函数
├── .gitignore          # Git 忽略文件配置
├── README.md           # 项目说明文档
├── go.mod              # Go 模块文件
├── go.sum              # Go 依赖校验文件
└── stable_diffusion.go # 根目录入口文件
```
注意：lib目录中所有动态链接库文件，需根据lib/version.txt版本到https://github.com/leejet/stable-diffusion.cpp/releases下载

## 🚀 快速开始

### 1. 安装依赖

```bash
go get github.com/Pendra-Cloud/stable-diffusion-go
```

### 2. 准备模型文件

使用前需要准备模型文件，支持多种格式：
- 扩散模型：`.gguf` 格式（如 z_image_turbo-Q4_K_M.gguf）
- LLM 模型：`.gguf` 格式（如 Qwen3-4B-Instruct-2507-Q4_K_M.gguf）
- VAE 模型：`.safetensors` 格式（如 diffusion_pytorch_model.safetensors）

### 3. 动态库说明

项目已包含多平台预编译动态库，位于 `pkg/sd/lib/` 目录下：
- **Windows**：提供多个版本以适应不同硬件
  - `avx/`：支持 AVX 指令集
  - `avx2/`：支持 AVX2 指令集
  - `avx512/`：支持 AVX512 指令集
  - `cuda12/`：支持 CUDA 12
  - `noavx/`：不依赖 AVX 指令集
  - `rocm/`：支持 ROCm
  - `vulkan/`：支持 Vulkan
- **Linux**：`libstable-diffusion.so`
- **macOS**：`libstable-diffusion.dylib`

程序会自动根据当前环境选择合适的动态库，无需手动指定。

### 4. 运行示例

#### 文本到图像生成

```bash
# 进入示例目录
cd examples

# 运行文本到图像示例
go run txt2img.go
```

示例代码：

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	fmt.Println("Stable Diffusion Go - Text to Image Example")
	fmt.Println("===============================================")

	// 创建 Stable Diffusion 实例
	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "path/to/diffusion_model.gguf",
		LLMPath:            "path/to/llm_model.gguf",
		VAEPath:            "path/to/vae_model.safetensors",
		DiffusionFlashAttn: true,
		OffloadParamsToCPU: true,
	})

	if err != nil {
		fmt.Println("创建实例失败:", err)
		return
	}
	defer sd.Free()

	// 生成图像
	err = sd.GenerateImage(&stablediffusion.ImgGenParams{
		Prompt:      "一位穿着明朝服饰的美女行走在花园中",
		Width:       512,
		Height:      512,
		SampleSteps: 10,
		CfgScale:    1.0,
	}, "output_demo.png")

	if err != nil {
		fmt.Println("生成图像失败:", err)
		return
	}

	fmt.Println("图像生成成功!")
}
```
![](output_demo.png)

#### 文本到视频生成

```bash
# 运行文本到视频示例
go run txt2vid.go
```

## 📚 核心功能

### 1. 上下文管理

- 创建和销毁 Stable Diffusion 上下文
- 支持多种模型路径配置
- 提供丰富的性能优化参数

### 2. 文本到图像生成 (txt2img)

- 根据文本描述生成高质量图像
- 支持中文和英文提示词
- 可调整图像尺寸、采样步数、CFG 比例等参数
- 支持随机种子生成

### 3. 文本到视频生成 (txt2vid)

- 根据文本提示生成视频
- 支持自定义帧数和分辨率
- 支持 Easycache 优化
- 集成 FFmpeg 实现视频编码

## 📝 使用指南

### 基本用法

1. **创建实例**：使用 `NewStableDiffusion` 创建 Stable Diffusion 实例
2. **配置参数**：设置上下文参数和生成参数
3. **生成内容**：调用 `GenerateImage` 或 `GenerateVideo` 生成内容
4. **释放资源**：使用 `defer sd.Free()` 释放资源

### 上下文参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| DiffusionModelPath | string | 扩散模型文件路径 |
| LLMPath | string | LLM 模型文件路径 |
| VAEPath | string | VAE 模型文件路径 |
| NThreads | int | 线程数 |
| DiffusionFlashAttn | bool | 是否启用 FlashAttention |
| OffloadParamsToCPU | bool | 是否将部分参数卸载到 CPU |
| WType | SDType | 模型量化类型 |

### 图像生成参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| Prompt | string | 提示词 |
| NegativePrompt | string | 负面提示词 |
| Width | int | 图像宽度 |
| Height | int | 图像高度 |
| Seed | int | 随机种子 |
| SampleSteps | int | 采样步数 |
| CfgScale | float64 | CFG 比例 |
| Strength | float64 | 初始图像强度（仅 img2img） |

## 🔧 性能优化

### 1. 调整线程数

根据 CPU 核心数调整 `NThreads` 参数：

```go
ctxParams := &stablediffusion.ContextParams{
    // 其他参数...
    NThreads: 8, // 根据 CPU 核心数调整
}
```

### 2. 使用量化模型

使用量化模型可以提高性能和减少内存占用：

```go
ctxParams := &stablediffusion.ContextParams{
    // 其他参数...
    WType: stablediffusion.SDTypeQ4_K, // 使用 Q4_K 量化模型
}
```

### 3. 调整采样步数

减少采样步数可以提高生成速度，但可能降低图像质量：

```go
imgGenParams := &stablediffusion.ImgGenParams{
    // 其他参数...
    SampleSteps: 10, // 减少采样步数
}
```

### 4. 启用 FlashAttention

启用 FlashAttention 可以加速扩散过程：

```go
ctxParams := &stablediffusion.ContextParams{
    // 其他参数...
    DiffusionFlashAttn: true,
}
```

## ⚠️ 注意事项

1. **动态库路径**：程序会自动从 `pkg/sd/lib/` 目录以及当前环境选择合适的动态库
2. **模型兼容性**：确保使用与 stable-diffusion.cpp 兼容的模型格式
3. **依赖项**：根据需要安装 CUDA或Vulkan 等依赖
4. **视频生成**：需要安装 FFmpeg 来编码视频
5. **内存占用**：大模型可能需要较多内存，建议使用量化模型
6. **关于AMD显卡（windows平台）**：若使用的是AMD显卡（包括AMD集成显卡），需下载ROCM的库放在项目根目录中，下载地址：https://github.com/leejet/stable-diffusion.cpp/releases/download/master-453-4ff2c8c/sd-master-4ff2c8c-bin-win-rocm-x64.zip
7. **关于Vulkan**：若使用的是非nvidia的显卡(如AMD或Inter的显卡，包括集成显卡)，可安装vulkan来启用GPU加速

## 📦 示例程序

### 文本到图像示例

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	// 创建实例
	sd, err := stablediffusion.NewStableDiffusion(&stablediffusion.ContextParams{
		DiffusionModelPath: "models/z_image_turbo-Q4_K_M.gguf",
		LLMPath:            "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
		VAEPath:            "models/diffusion_pytorch_model.safetensors",
		DiffusionFlashAttn: true,
	})
	if err != nil {
		fmt.Println("创建实例失败:", err)
		return
	}
	defer sd.Free()

	// 生成图像
	err = sd.GenerateImage(&stablediffusion.ImgGenParams{
		Prompt:      "一只可爱的柯基犬在草地上奔跑",
		Width:       512,
		Height:      512,
		SampleSteps: 15,
		CfgScale:    2.0,
	}, "output_corgi.png")

	if err != nil {
		fmt.Println("生成图像失败:", err)
		return
	}

	fmt.Println("图像生成成功！")
}
```

### 文本到视频示例

```go
package main

import (
	"fmt"
	stablediffusion "github.com/Pendra-Cloud/stable-diffusion-go"
)

func main() {
	// 创建实例
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

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 🔗 相关项目

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): 稳定扩散模型的 C++ 实现
- [purego](https://github.com/ebitengine/purego): 无需 cgo 的 Go 语言 FFI 库

## 📞 支持

如果您在使用过程中遇到问题，请：
1. 查看示例代码
2. 检查动态库路径和模型文件
3. 查看项目 Issues
4. 提交新的 Issue

---

感谢您使用 stable-diffusion-go！如果这个项目对您有帮助，请给我们一个 Star ⭐️
