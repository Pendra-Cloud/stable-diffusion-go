package stable_diffusion

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/Pendra-Cloud/stable-diffusion-go/pkg/sd"
)

// flowShiftOrAuto returns the caller's flow_shift, or the native "auto" sentinel
// (INFINITY) when it is left at zero. Upstream's sd_sample_params_init defaults
// flow_shift to INFINITY, which makes the library pick a model-specific shift
// (e.g. 5.0 / 3.0 / 1.5 depending on the model). Because this wrapper builds the
// sample-params struct as a literal rather than via sd_sample_params_init, we
// must reproduce that sentinel so a zero value doesn't silently force shift=0
// and degrade flow-model output.
func flowShiftOrAuto(v float32) float32 {
	if v == 0 {
		return float32(math.Inf(1))
	}
	return v
}

// Load loads the stable-diffusion shared library from libDir. An empty libDir
// falls back to the OS default search path. It must be called once before
// creating a context or generating; it is idempotent and returns an error
// (never panics) when the library is absent or incompatible.
func Load(libDir string) error {
	return sd.Load(libDir)
}

// Embedding embedding structure for defining model embeddings
type Embedding struct {
	Name string // Embedding name
	Path string // Embedding file path
}

// RNGTypeMap RNG type mapping
var RNGTypeMap = map[string]sd.RngType{
	"default":    sd.DefaultRNG,
	"cuda":       sd.CUDARNG, // Default
	"cpu":        sd.CPURNG,
	"type_count": sd.RNGTypeCount,
}

// SampleMethodMap sampling method mapping
var SampleMethodMap = map[string]sd.SampleMethod{
	"default":             -1, // Default
	"euler":               sd.EulerSampleMethod,
	"euler_a":             sd.EulerASampleMethod,
	"heun":                sd.HeunSampleMethod,
	"dpm2":                sd.DPM2SampleMethod,
	"dpm++2s_a":           sd.DPMPP2SASampleMethod,
	"dpm++2m":             sd.DPMPP2MSampleMethod,
	"dpm++2mv2":           sd.DPMPP2Mv2SampleMethod,
	"ipndm":               sd.IPNDMSampleMethod,
	"ipndm_v":             sd.IPNDMSampleMethodV,
	"lcm":                 sd.LCMSampleMethod,
	"ddim_trailing":       sd.DDIMTrailingSampleMethod,
	"tcd":                 sd.TCDSampleMethod,
	"res_multistep":       sd.ResMultistepSampleMethod,
	"res_2s":              sd.Res2SSampleMethod,
	"er_sde":              sd.ERSDESampleMethod,
	"euler_cfg_pp":        sd.EulerCFGPPSampleMethod,
	"euler_a_cfg_pp":      sd.EulerACFGPPSampleMethod,
	"euler_ge":            sd.EulerGESampleMethod,
	"sample_method_count": sd.SampleMethodCount,
}

// SchedulerMap scheduler mapping
var SchedulerMap = map[string]sd.Scheduler{
	"default":         -1, // Default
	"discrete":        sd.DiscreteScheduler,
	"karras":          sd.KarrasScheduler,
	"exponential":     sd.ExponentialScheduler,
	"ays":             sd.AYSScheduler,
	"gits":            sd.GITScheduler,
	"sgm_uniform":     sd.SGMUniformScheduler,
	"simple":          sd.SimpleScheduler,
	"smoothstep":      sd.SmoothstepScheduler,
	"kl_optimal":      sd.KLOptimalScheduler,
	"lcm":             sd.LCMScheduler,
	"bong_tangent":    sd.BongTangentScheduler,
	"ltx2":            sd.LTX2Scheduler,
	"scheduler_count": sd.SchedulerCount,
}

// PredictionMap prediction type mapping
var PredictionMap = map[string]sd.Prediction{
	"eps":        sd.EPSPred,
	"v":          sd.VPred,
	"edm_v":      sd.EDMVPred,
	"flow":       sd.FlowPred,
	"flux_flow":  sd.FluxFlowPred,
	"flux2_flow": sd.Flux2FlowPred,
	"default":    sd.PredictionCount, // Default
}

// SDTypeMap SDType mapping
var SDTypeMap = map[string]sd.SDType{
	"f32":  sd.SDTypeF32,
	"f16":  sd.SDTypeF16,
	"q4_0": sd.SDTypeQ4_0,
	"q4_1": sd.SDTypeQ4_1,
	"q5_0": sd.SDTypeQ5_0,
	"q5_1": sd.SDTypeQ5_1,
	"q8_0": sd.SDTypeQ8_0,
	"q8_1": sd.SDTypeQ8_1,
	// k-quantizations
	"q2_k":    sd.SDTypeQ2_K,
	"q3_k":    sd.SDTypeQ3_K,
	"q4_k":    sd.SDTypeQ4_K,
	"q5_k":    sd.SDTypeQ5_K,
	"q6_k":    sd.SDTypeQ6_K,
	"q8_k":    sd.SDTypeQ8_K,
	"iq2_xxs": sd.SDTypeIQ2_XXS,
	"iq2_xs":  sd.SDTypeIQ2_XS,
	"iq3_xxs": sd.SDTypeIQ3_XXS,
	"iq1_s":   sd.SDTypeIQ1_S,
	"iq4_nl":  sd.SDTypeIQ4_NL,
	"iq3_s":   sd.SDTypeIQ3_S,
	"iq2_s":   sd.SDTypeIQ2_S,
	"iq4_xs":  sd.SDTypeIQ4_XS,
	"i8":      sd.SDTypeI8,
	"i16":     sd.SDTypeI16,
	"i32":     sd.SDTypeI32,
	"i64":     sd.SDTypeI64,
	"f64":     sd.SDTypeF64,
	"iq1_m":   sd.SDTypeIQ1_M,
	"bf16":    sd.SDTypeBF16,
	// "q4_0_4_4": SD_TYPE_Q4_0_4_4,
	// "q4_0_4_8": SD_TYPE_Q4_0_4_8,
	// "q4_0_8_8": SD_TYPE_Q4_0_8_8,
	"tq1_0": sd.SDTypeTQ1_0,
	"tq2_0": sd.SDTypeTQ2_0,
	// "iq4_nl_4_4": SD_TYPE_IQ4_NL_4_4,
	// "iq4_nl_4_8": SD_TYPE_IQ4_NL_4_8,
	// "iq4_nl_8_8": SD_TYPE_IQ4_NL_8_8,
	"mxfp4":   sd.SDTypeMXFP4,
	"nvfp4":   sd.SDTypeNVFP4,
	"q1_0":    sd.SDTypeQ1_0,
	"default": sd.SDTypeCount, // Default
}

// PreviewMap preview type mapping
var PreviewMap = map[string]sd.Preview{
	"none":          sd.PreviewNone, // Default
	"proj":          sd.PreviewProj,
	"tae":           sd.PreviewTAE,
	"vae":           sd.PreviewVAE,
	"preview_count": sd.PreviewCount,
}

// LoraApplyModeMap LoRA apply mode mapping
var LoraApplyModeMap = map[string]sd.LoraApplyMode{
	"auto":                  sd.LoraApplyAuto, // Default
	"immediately":           sd.LoraApplyImmediately,
	"at_runtime":            sd.LoraApplyAtRuntime,
	"lora_apply_mode_count": sd.LoraApplyModeCount,
}

// VAEFormatMap VAE format mapping (controls how the VAE weights are interpreted)
var VAEFormatMap = map[string]sd.SDVAEFormat{
	"auto":  sd.VAEFormatAuto, // Default
	"flux":  sd.FluxVAEFormat,
	"sd3":   sd.SD3VAEFormat,
	"flux2": sd.Flux2VAEFormat,
}

// HiresUpscalerMap hi-res-fix upscaler mapping
var HiresUpscalerMap = map[string]sd.SDHiresUpscaler{
	"none":                       sd.HiresUpscalerNone, // Default
	"latent":                     sd.HiresUpscalerLatent,
	"latent_nearest":             sd.HiresUpscalerLatentNearest,
	"latent_nearest_exact":       sd.HiresUpscalerLatentNearestExact,
	"latent_antialiased":         sd.HiresUpscalerLatentAntialiased,
	"latent_bicubic":             sd.HiresUpscalerLatentBicubic,
	"latent_bicubic_antialiased": sd.HiresUpscalerLatentBicubicAntialiased,
	"lanczos":                    sd.HiresUpscalerLanczos,
	"nearest":                    sd.HiresUpscalerNearest,
	"model":                      sd.HiresUpscalerModel,
}

// ContextParams context parameters structure for initializing Stable Diffusion context
type ContextParams struct {
	ModelPath                   string     // Full model path
	ClipLPath                   string     // CLIP-L text encoder path
	ClipGPath                   string     // CLIP-G text encoder path
	ClipVisionPath              string     // CLIP Vision encoder path
	T5XXLPath                   string     // T5-XXL text encoder path
	LLMPath                     string     // LLM text encoder path (e.g., qwenvl2.5 for qwen-image, mistral-small3.2 for flux2)
	LLMVisionPath               string     // LLM Vision encoder path
	DiffusionModelPath          string     // Standalone diffusion model path
	HighNoiseDiffusionModelPath string     // Standalone high noise diffusion model path
	UncondDiffusionModelPath    string     // Standalone unconditional diffusion model path
	EmbeddingsConnectorsPath    string     // Embeddings connectors model path
	VAEPath                     string     // VAE model path
	AudioVAEPath                string     // Audio VAE model path (for audio-capable video models)
	TAESDPath                   string     // TAE-SD model path, uses Tiny AutoEncoder for fast decoding (low quality)
	ControlNetPath              string     // ControlNet model path
	Embeddings                  *Embedding // Embedding information
	EmbeddingCount              uint32     // Number of embeddings
	PhotoMakerPath              string     // PhotoMaker model path
	TensorTypeRules             string     // Weight type rules per tensor pattern (e.g., "^vae\.=f16,model\.=q8_0")
	VAEDecodeOnly               bool       // Process VAE using only decode mode
	FreeParamsImmediately       bool       // Whether to free parameters immediately
	NThreads                    int32      // Number of threads to use for generation
	WType                       string     // Weight type (default: auto-detect from model file)
	RNGType                     string     // Random number generator type (default: "cuda")
	SamplerRNGType              string     // Sampler random number generator type (default: "cuda")
	Prediction                  string     // Prediction type override
	LoraApplyMode               string     // LoRA application mode (default: "auto")
	OffloadParamsToCPU          bool       // Keep weights in RAM to save VRAM, auto-load to VRAM when needed
	EnableMmap                  bool       // Whether to enable memory mapping
	KeepClipOnCPU               bool       // Keep CLIP on CPU (for low VRAM)
	KeepControlNetOnCPU         bool       // Keep ControlNet on CPU (for low VRAM)
	KeepVAEOnCPU                bool       // Keep VAE on CPU (for low VRAM)
	FlashAttn                   bool       // Use Flash attention across the whole model (significantly reduces memory usage)
	DiffusionFlashAttn          bool       // Use Flash attention in diffusion model (significantly reduces memory usage)
	TAEPreviewOnly              bool       // Prevent decoding final image with taesd (for preview="tae")
	DiffusionConvDirect         bool       // Use Conv2d direct in diffusion model
	VAEConvDirect               bool       // Use Conv2d direct in VAE model (should improve performance)
	CircularX                   bool       // Enable circular padding on X axis
	CircularY                   bool       // Enable circular padding on Y axis
	ForceSDXLVAConvScale        bool       // Force conv scale on SDXL VAE
	ChromaUseDitMask            bool       // Whether Chroma uses DiT mask
	ChromaUseT5Mask             bool       // Whether Chroma uses T5 mask
	ChromaT5MaskPad             int32      // Chroma T5 mask padding size
	QwenImageZeroCondT          bool       // Qwen-image zero condition T parameter
	VAEFormat                   string     // VAE weight format override: "auto" (default), "flux", "sd3", "flux2"
	MaxVRAM                     float32    // GiB budget for graph-cut segmented param offload (0 = disabled, -1 = auto: free VRAM minus 1 GiB)
	StreamLayers                bool       // Stream model weights from CPU during generation (residency+prefetch on top of MaxVRAM; no effect unless MaxVRAM is set)
	Backend                     string     // Compute backend override (empty = library default)
	ParamsBackend               string     // Params/storage backend override (empty = library default)
}

// Lora LoRA structure for defining LoRA model parameters
type Lora struct {
	IsHighNoise bool    // Whether it's a high noise LoRA
	Multiplier  float32 // LoRA multiplier
	Path        string  // LoRA file path
}

// PMParams PhotoMaker parameters structure for defining PhotoMaker related parameters
type PMParams struct {
	IDImages      *sd.SDImage // ID images pointer
	IDImagesCount int32       // Number of ID images
	IDEmbedPath   string      // PhotoMaker v2 ID embedding path
	StyleStrength float32     // Strength to keep PhotoMaker input identity
}

// ImgGenParams image generation parameters structure for defining image generation related parameters
type ImgGenParams struct {
	Loras              *Lora             // LoRA parameters
	LoraCount          uint32            // Number of LoRAs
	Prompt             string            // Prompt to render
	NegativePrompt     string            // Negative prompt
	ClipSkip           int32             // Skip last layers of CLIP network (1 = no skip, 2 = skip one layer, <=0 = not specified)
	InitImagePath      string            // Initial image path for guidance
	RefImagesPath      []string          // Array of reference image paths for Flux Kontext models
	RefImagesCount     int32             // Number of reference images
	AutoResizeRefImage bool              // Whether to auto-resize reference images
	IncreaseRefIndex   bool              // Whether to auto-increase index based on reference image list order (starting from 1)
	MaskImagePath      string            // Inpainting mask image path
	Width              int32             // Image width (pixels)
	Height             int32             // Image height (pixels)
	CfgScale           float32           // Unconditional guidance scale.
	ImageCfgScale      float32           // Image guidance scale for inpaint or instruct-pix2pix models (default: same as `CfgScale`).
	DistilledGuidance  float32           // Distilled guidance scale for models with guidance input.
	SkipLayers         []int32           // Layers to skip for SLG steps (SLG will be enabled at step int([STEPS]x[START]) and disabled at int([STEPS]x[END])).
	SkipLayerStart     float32           // SLG enabling point.
	SkipLayerEnd       float32           // SLG disabling point.
	SlgScale           float32           // Skip layer guidance (SLG) scale, only for DiT models.
	Scheduler          string            // Denoiser sigma scheduler (default: discrete).
	SampleMethod       string            // Sampling method (default: euler for Flux/SD3/Wan, euler_a otherwise).
	SampleSteps        int32             // Number of sample steps.
	Eta                float32           // Eta in DDIM, only for DDIM and TCD.
	ShiftedTimestep    int32             // Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant.
	CustomSigmas       []float32         // Custom sigma values for the sampler, comma-separated (e.g. "14.61,7.8,3.5,0.0").
	Strength           float32           // Noise/denoise strength (range [0.0, 1.0])
	Seed               int64             // RNG seed (< 0 for random seed)
	BatchCount         int32             // Number of images to generate
	ControlImagePath   string            // Control condition image path for ControlNet
	ControlStrength    float32           // Strength to apply ControlNet
	PMParams           *PMParams         // PhotoMaker parameters
	VAETilingParams    sd.SDTilingParams // VAE tiling parameters for reducing memory usage
	CacheParams        sd.SDCacheParams  // Cache parameters for DiT models
	FlowShift          float32           // Shift value for flow models (e.g. SD3.x, Flux); 0 = library default
	ExtraSampleArgs    string            // Extra model-specific sampler arguments (advanced)

	// Hi-res fix: optionally run a second high-resolution refinement pass.
	HiresEnabled           bool      // Enable hi-res fix
	HiresUpscaler          string    // Hi-res upscaler (see HiresUpscalerMap, e.g. "latent", "model"); empty = "none"
	HiresModelPath         string    // Upscaler model path (for HiresUpscaler == "model")
	HiresScale             float32   // Upscale factor (used when target dimensions are unset)
	HiresTargetWidth       int32     // Explicit hi-res target width (overrides HiresScale)
	HiresTargetHeight      int32     // Explicit hi-res target height (overrides HiresScale)
	HiresSteps             int32     // Sample steps for the hi-res pass
	HiresDenoisingStrength float32   // Denoising strength for the hi-res pass
	HiresUpscaleTileSize   int32     // Tile size for the hi-res upscale
	HiresCustomSigmas      []float32 // Custom sigmas for the hi-res pass
}

// VidGenParams video generation parameters structure for defining video generation related parameters
type VidGenParams struct {
	Loras             *Lora    // LoRA parameters
	LoraCount         uint32   // Number of LoRAs
	Prompt            string   // Prompt to render
	NegativePrompt    string   // Negative prompt
	ClipSkip          int32    // Skip last layers of CLIP network (1 = no skip, 2 = skip one layer, <=0 = not specified)
	InitImagePath     string   // Initial image path for starting generation
	EndImagePath      string   // End image path for ending generation (required for flf2v)
	ControlFramesPath []string // Array of control frame image paths for video
	ControlFramesSize int32    // Control frame size
	Width             int32    // Video width (pixels)
	Height            int32    // Video height (pixels)

	CfgScale          float32   // Unconditional guidance scale.
	ImageCfgScale     float32   // Image guidance scale for inpaint or instruct-pix2pix models (default: same as `CfgScale`).
	DistilledGuidance float32   // Distilled guidance scale for models with guidance input.
	SkipLayers        []int32   // Layers to skip for SLG steps (SLG will be enabled at step int([STEPS]x[START]) and disabled at int([STEPS]x[END])).
	SkipLayerStart    float32   // SLG enabling point.
	SkipLayerEnd      float32   // SLG disabling point.
	SlgScale          float32   // Skip layer guidance (SLG) scale, only for DiT models.
	Scheduler         string    // Denoiser sigma scheduler (default: discrete).
	SampleMethod      string    // Sampling method (default: euler for Flux/SD3/Wan, euler_a otherwise).
	SampleSteps       int32     // Number of sample steps.
	Eta               float32   // Eta in DDIM, only for DDIM and TCD.
	ShiftedTimestep   int32     // Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant.
	CustomSigmas      []float32 // Custom sigma values for the sampler, comma-separated (e.g. "14.61,7.8,3.5,0.0").
	FlowShift         float32   // Shift value for flow models (e.g. SD3.x, Wan); 0 = library default.

	HighNoiseCfgScale          float32   // High noise diffusion model equivalent of `cfg_scale`.
	HighNoiseImageCfgScale     float32   // High noise diffusion model equivalent of `image_cfg_scale`.
	HighNoiseDistilledGuidance float32   // High noise diffusion model equivalent of `guidance`.
	HighNoiseSkipLayers        []int32   // High noise diffusion model equivalent of `skip_layers`.
	HighNoiseSkipLayerStart    float32   // High noise diffusion model equivalent of `skip_layer_start`.
	HighNoiseSkipLayerEnd      float32   // High noise diffusion model equivalent of `skip_layer_end`.
	HighNoiseSlgScale          float32   // High noise diffusion model equivalent of `slg_scale`.
	HighNoiseScheduler         string    // High noise diffusion model equivalent of `scheduler`.
	HighNoiseSampleMethod      string    // High noise diffusion model equivalent of `sample_method`.
	HighNoiseSampleSteps       int32     // High noise diffusion model equivalent of `sample_steps` (default: -1 = auto).
	HighNoiseEta               float32   // High noise diffusion model equivalent of `eta`.
	HighNoiseShiftedTimestep   int32     // Shift timestep for NitroFusion models, default: 0, recommended N for NitroSD-Realism around 250 and 500 for NitroSD-Vibrant.
	HighNoiseCustomSigmas      []float32 // Custom sigma values for the sampler, comma-separated (e.g. "14.61,7.8,3.5,0.0").
	HighNoiseFlowShift         float32   // High noise diffusion model equivalent of `FlowShift`.

	MOEBoundary  float32          // Timestep boundary for Wan2.2 MoE models
	Strength     float32          // Noise/denoise strength (range [0.0, 1.0])
	Seed         int64            // RNG seed (< 0 for random seed)
	VideoFrames  int32            // Number of video frames to generate
	VaceStrength float32          // Wan VACE strength
	CacheParams  sd.SDCacheParams // Cache parameters for DiT models
}

// StableDiffusion Stable Diffusion structure containing context pointer
type StableDiffusion struct {
	ctx *sd.SDContext // SD context pointer
}

// Free frees the stable diffusion context
func (sDiffusion *StableDiffusion) Free() {
	if sDiffusion.ctx != nil {
		sDiffusion.ctx.Free()
		sDiffusion.ctx = nil
	}
}

// NewStableDiffusion creates a stable diffusion instance
func NewStableDiffusion(ctxParams *ContextParams) (*StableDiffusion, error) {
	// 1. Initialize context parameters
	var sdCtxParams sd.SDContextParams
	sd.ContextParamsInit(&sdCtxParams)

	if ctxParams.ModelPath != "" {
		sdCtxParams.ModelPath = sd.CString(ctxParams.ModelPath)
	}

	if ctxParams.ClipLPath != "" {
		sdCtxParams.ClipLPath = sd.CString(ctxParams.ClipLPath)
	}

	if ctxParams.ClipGPath != "" {
		sdCtxParams.ClipGPath = sd.CString(ctxParams.ClipGPath)
	}

	if ctxParams.ClipVisionPath != "" {
		sdCtxParams.ClipVisionPath = sd.CString(ctxParams.ClipVisionPath)
	}

	if ctxParams.T5XXLPath != "" {
		sdCtxParams.T5XXLPath = sd.CString(ctxParams.T5XXLPath)
	}

	if ctxParams.LLMPath != "" {
		sdCtxParams.LLMPath = sd.CString(ctxParams.LLMPath)
	}

	if ctxParams.LLMVisionPath != "" {
		sdCtxParams.LLMVisionPath = sd.CString(ctxParams.LLMVisionPath)
	}

	if ctxParams.DiffusionModelPath != "" {
		sdCtxParams.DiffusionModelPath = sd.CString(ctxParams.DiffusionModelPath)
	}

	if ctxParams.HighNoiseDiffusionModelPath != "" {
		sdCtxParams.HighNoiseDiffusionModelPath = sd.CString(ctxParams.HighNoiseDiffusionModelPath)
	}

	if ctxParams.UncondDiffusionModelPath != "" {
		sdCtxParams.UncondDiffusionModelPath = sd.CString(ctxParams.UncondDiffusionModelPath)
	}

	if ctxParams.EmbeddingsConnectorsPath != "" {
		sdCtxParams.EmbeddingsConnectorsPath = sd.CString(ctxParams.EmbeddingsConnectorsPath)
	}

	if ctxParams.VAEPath != "" {
		sdCtxParams.VAEPath = sd.CString(ctxParams.VAEPath)
	}

	if ctxParams.AudioVAEPath != "" {
		sdCtxParams.AudioVAEPath = sd.CString(ctxParams.AudioVAEPath)
	}

	if ctxParams.TAESDPath != "" {
		sdCtxParams.TAESDPath = sd.CString(ctxParams.TAESDPath)
	}

	if ctxParams.ControlNetPath != "" {
		sdCtxParams.ControlNetPath = sd.CString(ctxParams.ControlNetPath)
	}

	if ctxParams.Embeddings != nil {
		sdCtxParams.Embeddings = &sd.SDEmbedding{
			Name: sd.CString(ctxParams.Embeddings.Name),
			Path: sd.CString(ctxParams.Embeddings.Path),
		}
	}

	if ctxParams.EmbeddingCount > 0 {
		sdCtxParams.EmbeddingCount = ctxParams.EmbeddingCount
	}

	if ctxParams.PhotoMakerPath != "" {
		sdCtxParams.PhotoMakerPath = sd.CString(ctxParams.PhotoMakerPath)
	}

	if ctxParams.TensorTypeRules != "" {
		sdCtxParams.TensorTypeRules = sd.CString(ctxParams.TensorTypeRules)
	}

	sdCtxParams.VAEDecodeOnly = ctxParams.VAEDecodeOnly
	sdCtxParams.FreeParamsImmediately = ctxParams.FreeParamsImmediately

	if ctxParams.NThreads > 0 {
		sdCtxParams.NThreads = ctxParams.NThreads
	}

	if ctxParams.WType != "" {
		if WType, ok := SDTypeMap[ctxParams.WType]; ok {
			sdCtxParams.WType = WType
		} else {
			return nil, fmt.Errorf("Invalid WType: %s", ctxParams.WType)
		}
	}

	if ctxParams.RNGType != "" {
		if RNGType, ok := RNGTypeMap[ctxParams.RNGType]; ok {
			sdCtxParams.RNGType = RNGType
		} else {
			return nil, fmt.Errorf("Invalid RNG type: %s", ctxParams.RNGType)
		}
	}

	if ctxParams.SamplerRNGType != "" {
		if RNGType, ok := RNGTypeMap[ctxParams.SamplerRNGType]; ok {
			sdCtxParams.SamplerRNGType = RNGType
		} else {
			return nil, fmt.Errorf("Invalid Sampler RNG type: %s", ctxParams.SamplerRNGType)
		}
	}

	if ctxParams.Prediction != "" {
		if Prediction, ok := PredictionMap[ctxParams.Prediction]; ok {
			sdCtxParams.Prediction = Prediction
		} else {
			return nil, fmt.Errorf("Invalid Prediction: %s", ctxParams.Prediction)
		}
	}

	if ctxParams.LoraApplyMode != "" {
		if LoraApplyMode, ok := LoraApplyModeMap[ctxParams.LoraApplyMode]; ok {
			sdCtxParams.LoraApplyMode = LoraApplyMode
		} else {
			return nil, fmt.Errorf("Invalid LoraApplyMode: %s", ctxParams.LoraApplyMode)
		}
	}

	sdCtxParams.OffloadParamsToCPU = ctxParams.OffloadParamsToCPU
	sdCtxParams.EnableMmap = ctxParams.EnableMmap
	sdCtxParams.KeepClipOnCPU = ctxParams.KeepClipOnCPU
	sdCtxParams.KeepControlNetOnCPU = ctxParams.KeepControlNetOnCPU
	sdCtxParams.KeepVAEOnCPU = ctxParams.KeepVAEOnCPU
	sdCtxParams.FlashAttn = ctxParams.FlashAttn
	sdCtxParams.DiffusionFlashAttn = ctxParams.DiffusionFlashAttn
	sdCtxParams.TAEPreviewOnly = ctxParams.TAEPreviewOnly
	sdCtxParams.DiffusionConvDirect = ctxParams.DiffusionConvDirect
	sdCtxParams.VAEConvDirect = ctxParams.VAEConvDirect
	sdCtxParams.CircularX = ctxParams.CircularX
	sdCtxParams.CircularY = ctxParams.CircularY
	sdCtxParams.ForceSDXLVAConvScale = ctxParams.ForceSDXLVAConvScale
	sdCtxParams.ChromaUseDitMask = ctxParams.ChromaUseDitMask
	sdCtxParams.ChromaUseT5Mask = ctxParams.ChromaUseT5Mask

	if ctxParams.ChromaT5MaskPad != 0 {
		sdCtxParams.ChromaT5MaskPad = ctxParams.ChromaT5MaskPad
	}

	sdCtxParams.QwenImageZeroCondT = ctxParams.QwenImageZeroCondT

	if ctxParams.VAEFormat != "" {
		if vaeFormat, ok := VAEFormatMap[ctxParams.VAEFormat]; ok {
			sdCtxParams.VAEFormat = vaeFormat
		} else {
			return nil, fmt.Errorf("Invalid VAEFormat: %s", ctxParams.VAEFormat)
		}
	}

	sdCtxParams.MaxVRAM = ctxParams.MaxVRAM
	sdCtxParams.StreamLayers = ctxParams.StreamLayers

	if ctxParams.Backend != "" {
		sdCtxParams.Backend = sd.CString(ctxParams.Backend)
	}
	if ctxParams.ParamsBackend != "" {
		sdCtxParams.ParamsBackend = sd.CString(ctxParams.ParamsBackend)
	}

	// 2. Create new context
	ctx := sd.NewContext(&sdCtxParams)
	if ctx == nil {
		return nil, errors.New("failed to create context")
	}

	return &StableDiffusion{
		ctx: ctx,
	}, nil
}

// GenerateImage generates image from text or image
func (sDiffusion *StableDiffusion) GenerateImage(imgGenParams *ImgGenParams, newImagePath string) error {
	// Extract the directory part of newImagePath and create it if it doesn't exist
	dir := filepath.Dir(newImagePath)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err = os.MkdirAll(dir, os.ModePerm)
		if err != nil {
			return errors.New("failed to create directory")
		}
	}

	// Initialize image generation parameters
	var sdImgGenParams sd.SDImgGenParams
	sd.ImgGenParamsInit(&sdImgGenParams)

	// Set generation parameters
	sdImgGenParams.Prompt = sd.CString(imgGenParams.Prompt)
	if imgGenParams.NegativePrompt == "" {
		imgGenParams.NegativePrompt = "blurry, low quality, distorted"
	}

	sdImgGenParams.NegativePrompt = sd.CString(imgGenParams.NegativePrompt)

	if imgGenParams.Loras == nil {
		sdImgGenParams.Loras = &sd.SDLora{
			IsHighNoise: false,
			Multiplier:  0,
			Path:        sd.CString(""),
		}
	} else {
		sdImgGenParams.Loras = &sd.SDLora{
			IsHighNoise: imgGenParams.Loras.IsHighNoise,
			Multiplier:  imgGenParams.Loras.Multiplier,
			Path:        sd.CString(imgGenParams.Loras.Path),
		}
	}

	sdImgGenParams.LoraCount = imgGenParams.LoraCount

	if imgGenParams.ClipSkip == 0 {
		imgGenParams.ClipSkip = -1
	}
	sdImgGenParams.ClipSkip = imgGenParams.ClipSkip

	sdImgGenParams.InitImage = sd.GenerateImageFromPath(imgGenParams.InitImagePath)
	sdImgGenParams.RefImages = sd.GenerateImagesFromPaths(imgGenParams.RefImagesPath)
	// Set reference images count, use actual loaded count if user didn't provide
	sdImgGenParams.RefImagesCount = int32(len(imgGenParams.RefImagesPath))
	if imgGenParams.RefImagesCount > 0 {
		sdImgGenParams.RefImagesCount = imgGenParams.RefImagesCount
	}
	sdImgGenParams.AutoResizeRefImage = imgGenParams.AutoResizeRefImage
	sdImgGenParams.IncreaseRefIndex = imgGenParams.IncreaseRefIndex
	sdImgGenParams.MaskImage = sd.GenerateImageFromPath(imgGenParams.MaskImagePath)

	// For img2img, ensure generated image dimensions match initial image dimensions
	// If initial image is specified, use initial image dimensions
	if imgGenParams.InitImagePath != "" {
		// Use initial image dimensions and convert type
		sdImgGenParams.Width = int32(sdImgGenParams.InitImage.Width)
		sdImgGenParams.Height = int32(sdImgGenParams.InitImage.Height)
	} else {
		// Otherwise use default dimensions
		if imgGenParams.Width == 0 {
			imgGenParams.Width = 512
		}
		if imgGenParams.Height == 0 {
			imgGenParams.Height = 512
		}
		sdImgGenParams.Width = imgGenParams.Width
		sdImgGenParams.Height = imgGenParams.Height
	}

	if imgGenParams.CfgScale == 0 {
		imgGenParams.CfgScale = 5.0
	}
	if imgGenParams.ImageCfgScale == 0 {
		imgGenParams.ImageCfgScale = 1.0
	}

	if imgGenParams.DistilledGuidance == 0 {
		imgGenParams.DistilledGuidance = 3.5
	}

	var skipLayersPtr *int32
	if len(imgGenParams.SkipLayers) > 0 {
		skipLayersPtr = &imgGenParams.SkipLayers[0]
	} else {
		skipLayersPtr = nil
	}

	if imgGenParams.SkipLayerStart == 0 {
		imgGenParams.SkipLayerStart = 0.01
	}
	if imgGenParams.SkipLayerEnd == 0 {
		imgGenParams.SkipLayerEnd = 0.2
	}

	var defaultSampleMethod sd.SampleMethod
	if imgGenParams.SampleMethod == "" || imgGenParams.SampleMethod == "default" {
		defaultSampleMethod = sDiffusion.ctx.GetDefaultSampleMethod()
	} else {
		if sampleMethod, ok := SampleMethodMap[imgGenParams.SampleMethod]; ok {
			defaultSampleMethod = sampleMethod
		} else {
			return fmt.Errorf("Invalid SampleMethod: %s", imgGenParams.SampleMethod)
		}
	}

	var defaultScheduler sd.Scheduler
	if imgGenParams.Scheduler == "" || imgGenParams.Scheduler == "default" {
		defaultScheduler = sDiffusion.ctx.GetDefaultScheduler(defaultSampleMethod)
	} else {
		if scheduler, ok := SchedulerMap[imgGenParams.Scheduler]; ok {
			defaultScheduler = scheduler
		} else {
			return fmt.Errorf("Invalid Scheduler: %s", imgGenParams.Scheduler)
		}
	}

	if imgGenParams.SampleSteps == 0 {
		imgGenParams.SampleSteps = 20
	}

	if imgGenParams.Eta == 0 {
		imgGenParams.Eta = 1.0
	}

	var customSigmasPtr *float32
	if len(imgGenParams.CustomSigmas) > 0 {
		customSigmasPtr = &imgGenParams.CustomSigmas[0]
	} else {
		customSigmasPtr = nil
	}

	sdImgGenParams.SampleParams = sd.SDSampleParams{
		Guidance: sd.SDGuidanceParams{
			TxtCfg:            imgGenParams.CfgScale,
			ImgCfg:            imgGenParams.ImageCfgScale,
			DistilledGuidance: imgGenParams.DistilledGuidance,
			SLG: sd.SDSLGParams{
				Layers:     skipLayersPtr,
				LayerCount: uintptr(len(imgGenParams.SkipLayers)),
				LayerStart: imgGenParams.SkipLayerStart,
				LayerEnd:   imgGenParams.SkipLayerEnd,
				Scale:      imgGenParams.SlgScale,
			},
		},
		SampleMethod:      defaultSampleMethod,
		Scheduler:         defaultScheduler,
		SampleSteps:       imgGenParams.SampleSteps,
		Eta:               imgGenParams.Eta,
		ShiftedTimestep:   imgGenParams.ShiftedTimestep,
		CustomSigmas:      customSigmasPtr,
		CustomSigmasCount: int32(len(imgGenParams.CustomSigmas)),
		FlowShift:         flowShiftOrAuto(imgGenParams.FlowShift),
		ExtraSampleArgs:   sd.CString(imgGenParams.ExtraSampleArgs),
	}

	if imgGenParams.Strength == 0 {
		imgGenParams.Strength = 0.75
	}
	sdImgGenParams.Strength = imgGenParams.Strength

	if imgGenParams.Seed == 0 {
		imgGenParams.Seed = 42
	}
	sdImgGenParams.Seed = imgGenParams.Seed

	if imgGenParams.BatchCount == 0 {
		imgGenParams.BatchCount = 1
	}
	sdImgGenParams.BatchCount = imgGenParams.BatchCount

	sdImgGenParams.ControlImage = sd.GenerateImageFromPath(imgGenParams.ControlImagePath)

	if imgGenParams.ControlStrength == 0 {
		imgGenParams.ControlStrength = 0.9
	}
	sdImgGenParams.ControlStrength = imgGenParams.ControlStrength

	if imgGenParams.PMParams != nil {
		sdImgGenParams.PMParams = sd.SDPMParams{
			IDImages:      imgGenParams.PMParams.IDImages,
			IDImagesCount: imgGenParams.PMParams.IDImagesCount,
			IDEmbedPath:   sd.CString(imgGenParams.PMParams.IDEmbedPath),
			StyleStrength: imgGenParams.PMParams.StyleStrength,
		}
	}

	if imgGenParams.VAETilingParams != (sd.SDTilingParams{}) {
		sdImgGenParams.VAETilingParams = imgGenParams.VAETilingParams
	} else {
		sdImgGenParams.VAETilingParams = sd.SDTilingParams{
			Enabled:       false,
			TileSizeX:     0,
			TileSizeY:     0,
			TargetOverlap: 0.5,
			RelSizeX:      0,
			RelSizeY:      0,
		}
	}

	// Initialize cache parameters
	var cacheParams sd.SDCacheParams
	sd.CacheParamsInit(&cacheParams)

	// If user provided cache parameters, use them
	if imgGenParams.CacheParams != (sd.SDCacheParams{}) {
		sdImgGenParams.Cache = imgGenParams.CacheParams
	} else {
		// Otherwise use default parameters
		sdImgGenParams.Cache = cacheParams
		// Set some reasonable default values
		sdImgGenParams.Cache.Mode = sd.SDCacheDisabled
		sdImgGenParams.Cache.ReuseThreshold = 0.2
		sdImgGenParams.Cache.StartPercent = 0.15
		sdImgGenParams.Cache.EndPercent = 0.95
	}

	// Hi-res fix: sd_img_gen_params_init already populated sensible (disabled)
	// Hires defaults; only override when the caller opts in, and only for the
	// fields they actually set so the library defaults survive for the rest.
	if imgGenParams.HiresEnabled {
		sdImgGenParams.Hires.Enabled = true
		if imgGenParams.HiresUpscaler != "" {
			if upscaler, ok := HiresUpscalerMap[imgGenParams.HiresUpscaler]; ok {
				sdImgGenParams.Hires.Upscaler = upscaler
			} else {
				return fmt.Errorf("Invalid HiresUpscaler: %s", imgGenParams.HiresUpscaler)
			}
		}
		if imgGenParams.HiresModelPath != "" {
			sdImgGenParams.Hires.ModelPath = sd.CString(imgGenParams.HiresModelPath)
		}
		if imgGenParams.HiresScale != 0 {
			sdImgGenParams.Hires.Scale = imgGenParams.HiresScale
		}
		if imgGenParams.HiresTargetWidth != 0 {
			sdImgGenParams.Hires.TargetWidth = imgGenParams.HiresTargetWidth
		}
		if imgGenParams.HiresTargetHeight != 0 {
			sdImgGenParams.Hires.TargetHeight = imgGenParams.HiresTargetHeight
		}
		if imgGenParams.HiresSteps != 0 {
			sdImgGenParams.Hires.Steps = imgGenParams.HiresSteps
		}
		if imgGenParams.HiresDenoisingStrength != 0 {
			sdImgGenParams.Hires.DenoisingStrength = imgGenParams.HiresDenoisingStrength
		}
		if imgGenParams.HiresUpscaleTileSize != 0 {
			sdImgGenParams.Hires.UpscaleTileSize = imgGenParams.HiresUpscaleTileSize
		}
		if len(imgGenParams.HiresCustomSigmas) > 0 {
			sdImgGenParams.Hires.CustomSigmas = &imgGenParams.HiresCustomSigmas[0]
			sdImgGenParams.Hires.CustomSigmasCount = int32(len(imgGenParams.HiresCustomSigmas))
		}
	}

	// Generate image. The context is intentionally NOT freed here: the model is
	// loaded once in NewStableDiffusion and reused across many GenerateImage
	// calls. Teardown is the caller's responsibility via (*StableDiffusion).Free.
	img := sDiffusion.ctx.GenerateImage(&sdImgGenParams)
	if img == nil {
		return errors.New("failed to generate image")
	}
	// generate_image returns a malloc'd array of BatchCount images, each with a
	// malloc'd pixel buffer; free it once we're done with it to avoid a leak.
	defer sd.FreeImages(img, int(sdImgGenParams.BatchCount))

	fmt.Println("\nImage generated successfully!")
	fmt.Printf("Image dimensions: %dx%d\n", img.Width, img.Height)
	fmt.Printf("Channels: %d\n", img.Channel)
	fmt.Printf("Data pointer: %p\n", unsafe.Pointer(img.Data))

	// Save image
	err := sd.SaveImage(img, newImagePath)
	if err != nil {
		return errors.New("failed to save image")
	}

	return nil

}

// GenerateVideo generates video
func (sDiffusion *StableDiffusion) GenerateVideo(vidGenParams *VidGenParams, newVideoPath string) error {
	// Extract the directory part of newVideoPath and create it if it doesn't exist
	dir := filepath.Dir(newVideoPath)
	tmpDir := filepath.Join(dir, "tmp")
	if _, err := os.Stat(tmpDir); os.IsNotExist(err) {
		err = os.MkdirAll(tmpDir, os.ModePerm)
		if err != nil {
			return errors.New("failed to create directory")
		}
	}

	// Initialize video generation parameters
	var sdVidGenParams sd.SDVidGenParams
	sd.VidGenParamsInit(&sdVidGenParams)

	// Set generation parameters
	sdVidGenParams.Prompt = sd.CString(vidGenParams.Prompt)
	sdVidGenParams.NegativePrompt = sd.CString(vidGenParams.NegativePrompt)

	if vidGenParams.Loras == nil {
		sdVidGenParams.Loras = &sd.SDLora{
			IsHighNoise: false,
			Multiplier:  0,
			Path:        sd.CString(""),
		}
	} else {
		sdVidGenParams.Loras = &sd.SDLora{
			IsHighNoise: vidGenParams.Loras.IsHighNoise,
			Multiplier:  vidGenParams.Loras.Multiplier,
			Path:        sd.CString(vidGenParams.Loras.Path),
		}
	}

	sdVidGenParams.LoraCount = vidGenParams.LoraCount

	if vidGenParams.ClipSkip == 0 {
		vidGenParams.ClipSkip = -1
	}
	sdVidGenParams.ClipSkip = vidGenParams.ClipSkip

	sdVidGenParams.InitImage = sd.GenerateImageFromPath(vidGenParams.InitImagePath)
	sdVidGenParams.EndImage = sd.GenerateImageFromPath(vidGenParams.EndImagePath)

	// Process control frames
	var controlFrames []sd.SDImage
	if len(vidGenParams.ControlFramesPath) > 0 {
		controlFrames = make([]sd.SDImage, len(vidGenParams.ControlFramesPath))
		for i, path := range vidGenParams.ControlFramesPath {
			controlFrames[i] = sd.GenerateImageFromPath(path)
		}
	}
	if len(controlFrames) > 0 {
		sdVidGenParams.ControlFrames = &controlFrames[0]
		sdVidGenParams.ControlFramesSize = int32(len(controlFrames))
	}

	if vidGenParams.Width == 0 {
		vidGenParams.Width = 512
	}
	if vidGenParams.Height == 0 {
		vidGenParams.Height = 512
	}
	sdVidGenParams.Width = vidGenParams.Width
	sdVidGenParams.Height = vidGenParams.Height

	if vidGenParams.CfgScale == 0 {
		vidGenParams.CfgScale = 6.0
	}
	if vidGenParams.ImageCfgScale == 0 {
		vidGenParams.ImageCfgScale = 1.0
	}

	if vidGenParams.DistilledGuidance == 0 {
		vidGenParams.DistilledGuidance = 3.5
	}

	var skipLayersPtr *int32
	if len(vidGenParams.SkipLayers) > 0 {
		skipLayersPtr = &vidGenParams.SkipLayers[0]
	} else {
		skipLayersPtr = nil
	}

	if vidGenParams.SkipLayerStart == 0 {
		vidGenParams.SkipLayerStart = 0.01
	}
	if vidGenParams.SkipLayerEnd == 0 {
		vidGenParams.SkipLayerEnd = 0.2
	}

	var defaultSampleMethod sd.SampleMethod
	if vidGenParams.SampleMethod == "" || vidGenParams.SampleMethod == "default" {
		defaultSampleMethod = sDiffusion.ctx.GetDefaultSampleMethod()
	} else {
		if sampleMethod, ok := SampleMethodMap[vidGenParams.SampleMethod]; ok {
			defaultSampleMethod = sampleMethod
		} else {
			return fmt.Errorf("Invalid SampleMethod: %s", vidGenParams.SampleMethod)
		}
	}

	var defaultScheduler sd.Scheduler
	if vidGenParams.Scheduler == "" || vidGenParams.Scheduler == "default" {
		defaultScheduler = sDiffusion.ctx.GetDefaultScheduler(defaultSampleMethod)
	} else {
		if scheduler, ok := SchedulerMap[vidGenParams.Scheduler]; ok {
			defaultScheduler = scheduler
		} else {
			return fmt.Errorf("Invalid Scheduler: %s", vidGenParams.Scheduler)
		}
	}

	if vidGenParams.SampleSteps == 0 {
		vidGenParams.SampleSteps = 20
	}

	if vidGenParams.Eta == 0 {
		vidGenParams.Eta = 1.0
	}

	var customSigmasPtr *float32
	if len(vidGenParams.CustomSigmas) > 0 {
		customSigmasPtr = &vidGenParams.CustomSigmas[0]
	} else {
		customSigmasPtr = nil
	}

	sdVidGenParams.SampleParams = sd.SDSampleParams{
		Guidance: sd.SDGuidanceParams{
			TxtCfg:            vidGenParams.CfgScale,
			ImgCfg:            vidGenParams.ImageCfgScale,
			DistilledGuidance: vidGenParams.DistilledGuidance,
			SLG: sd.SDSLGParams{
				Layers:     skipLayersPtr,
				LayerCount: uintptr(len(vidGenParams.SkipLayers)),
				LayerStart: vidGenParams.SkipLayerStart,
				LayerEnd:   vidGenParams.SkipLayerEnd,
				Scale:      vidGenParams.SlgScale,
			},
		},
		SampleMethod:      defaultSampleMethod,
		Scheduler:         defaultScheduler,
		SampleSteps:       vidGenParams.SampleSteps,
		Eta:               vidGenParams.Eta,
		ShiftedTimestep:   vidGenParams.ShiftedTimestep,
		CustomSigmas:      customSigmasPtr,
		CustomSigmasCount: int32(len(vidGenParams.CustomSigmas)),
		FlowShift:         flowShiftOrAuto(vidGenParams.FlowShift),
	}

	if vidGenParams.HighNoiseCfgScale == 0 {
		vidGenParams.HighNoiseCfgScale = 6.0
	}
	if vidGenParams.HighNoiseImageCfgScale == 0 {
		vidGenParams.HighNoiseImageCfgScale = 1.0
	}

	if vidGenParams.HighNoiseDistilledGuidance == 0 {
		vidGenParams.HighNoiseDistilledGuidance = 3.5
	}

	var highNoiseSkipLayersPtr *int32
	if len(vidGenParams.HighNoiseSkipLayers) > 0 {
		highNoiseSkipLayersPtr = &vidGenParams.HighNoiseSkipLayers[0]
	} else {
		highNoiseSkipLayersPtr = nil
	}

	if vidGenParams.HighNoiseSkipLayerStart == 0 {
		vidGenParams.HighNoiseSkipLayerStart = 0.01
	}
	if vidGenParams.HighNoiseSkipLayerEnd == 0 {
		vidGenParams.HighNoiseSkipLayerEnd = 0.2
	}

	var defaultHighNoiseSampleMethod sd.SampleMethod
	if vidGenParams.HighNoiseSampleMethod == "" || vidGenParams.HighNoiseSampleMethod == "default" {
		defaultHighNoiseSampleMethod = sDiffusion.ctx.GetDefaultSampleMethod()
	} else {
		if sampleMethod, ok := SampleMethodMap[vidGenParams.HighNoiseSampleMethod]; ok {
			defaultHighNoiseSampleMethod = sampleMethod
		} else {
			return fmt.Errorf("Invalid SampleMethod: %s", vidGenParams.HighNoiseSampleMethod)
		}
	}

	var defaultHighNoiseScheduler sd.Scheduler
	if vidGenParams.HighNoiseScheduler == "" || vidGenParams.HighNoiseScheduler == "default" {
		defaultHighNoiseScheduler = sDiffusion.ctx.GetDefaultScheduler(defaultHighNoiseSampleMethod)
	} else {
		if scheduler, ok := SchedulerMap[vidGenParams.Scheduler]; ok {
			defaultHighNoiseScheduler = scheduler
		} else {
			return fmt.Errorf("Invalid Scheduler: %s", vidGenParams.HighNoiseScheduler)
		}
	}

	if vidGenParams.HighNoiseSampleSteps == 0 {
		vidGenParams.HighNoiseSampleSteps = -1
	}

	if vidGenParams.HighNoiseEta == 0 {
		vidGenParams.HighNoiseEta = 1.0
	}

	var highNoiseCustomSigmasPtr *float32
	if len(vidGenParams.HighNoiseCustomSigmas) > 0 {
		highNoiseCustomSigmasPtr = &vidGenParams.HighNoiseCustomSigmas[0]
	} else {
		highNoiseCustomSigmasPtr = nil
	}

	sdVidGenParams.HighNoiseSampleParams = sd.SDSampleParams{
		Guidance: sd.SDGuidanceParams{
			TxtCfg:            vidGenParams.HighNoiseCfgScale,
			ImgCfg:            vidGenParams.HighNoiseImageCfgScale,
			DistilledGuidance: vidGenParams.HighNoiseDistilledGuidance,
			SLG: sd.SDSLGParams{
				Layers:     highNoiseSkipLayersPtr,
				LayerCount: uintptr(len(vidGenParams.HighNoiseSkipLayers)),
				LayerStart: vidGenParams.HighNoiseSkipLayerStart,
				LayerEnd:   vidGenParams.HighNoiseSkipLayerEnd,
				Scale:      vidGenParams.HighNoiseSlgScale,
			},
		},
		SampleMethod:      defaultHighNoiseSampleMethod,
		Scheduler:         defaultHighNoiseScheduler,
		SampleSteps:       vidGenParams.HighNoiseSampleSteps,
		Eta:               vidGenParams.HighNoiseEta,
		ShiftedTimestep:   vidGenParams.HighNoiseShiftedTimestep,
		CustomSigmas:      highNoiseCustomSigmasPtr,
		CustomSigmasCount: int32(len(vidGenParams.HighNoiseCustomSigmas)),
		FlowShift:         flowShiftOrAuto(vidGenParams.HighNoiseFlowShift),
	}

	if vidGenParams.MOEBoundary == 0 {
		vidGenParams.MOEBoundary = 0.875
	}
	sdVidGenParams.MOEBoundary = vidGenParams.MOEBoundary

	if vidGenParams.Strength == 0 {
		vidGenParams.Strength = 0.75
	}
	sdVidGenParams.Strength = vidGenParams.Strength

	if vidGenParams.Seed == 0 {
		vidGenParams.Seed = 42
	}
	sdVidGenParams.Seed = vidGenParams.Seed

	if vidGenParams.VideoFrames == 0 {
		vidGenParams.VideoFrames = 1
	}
	sdVidGenParams.VideoFrames = vidGenParams.VideoFrames

	if vidGenParams.VaceStrength == 0 {
		vidGenParams.VaceStrength = 1
	}
	sdVidGenParams.VaceStrength = vidGenParams.VaceStrength

	// Initialize cache parameters
	var cacheParams sd.SDCacheParams
	sd.CacheParamsInit(&cacheParams)

	// If user provided cache parameters, use them
	if vidGenParams.CacheParams != (sd.SDCacheParams{}) {
		sdVidGenParams.Cache = vidGenParams.CacheParams
	} else {
		// Otherwise use default parameters
		sdVidGenParams.Cache = cacheParams
		// Set some reasonable default values
		sdVidGenParams.Cache.Mode = sd.SDCacheDisabled
		sdVidGenParams.Cache.ReuseThreshold = 0.2
		sdVidGenParams.Cache.StartPercent = 0.15
		sdVidGenParams.Cache.EndPercent = 0.95
	}

	// Generate video
	frames, numFrames := sDiffusion.ctx.GenerateVideo(&sdVidGenParams)
	if frames == nil || numFrames == 0 {
		return errors.New("failed to generate video")
	}
	defer func() {
		sDiffusion.ctx.Free()
	}()

	// Save video frames
	if err := sd.SaveFrames(frames, tmpDir); err != nil {
		return errors.New("failed to save video")
	}
	if err := sd.EncodeVideo(tmpDir, newVideoPath, 30); err != nil {
		return errors.New("failed to encode video")
	}
	if err := sd.CleanupTempDir(tmpDir); err != nil {
		return errors.New("failed to cleanup temp directory")
	}

	return nil

}

type UpscalerParams struct {
	EsrganPath         string // ESRGAN model path
	OffloadParamsToCPU bool   // Whether to save parameters to CPU
	Direct             bool   // Whether to use direct mode
	NThreads           int    // Number of threads to use
	TileSize           int    // Tile size
	Backend            string // Compute backend override (empty = library default)
	ParamsBackend      string // Params/storage backend override (empty = library default)
}

type Upscaler struct {
	ctx *sd.UpscalerContext
}

// NewUpscaler creates a new upscaler context
func NewUpscaler(params *UpscalerParams) *Upscaler {
	if params.NThreads == 0 {
		params.NThreads = -1
	}

	if params.TileSize == 0 {
		params.TileSize = 128
	}

	ctx := sd.NewUpscalerContext(
		params.EsrganPath,
		params.OffloadParamsToCPU,
		params.Direct,
		params.NThreads,
		params.TileSize,
		params.Backend,
		params.ParamsBackend,
	)
	return &Upscaler{ctx: ctx}
}

// Upscale upscaling function
func (us *Upscaler) Upscale(inputImagePath string, upscaleFactor uint32, outputImagePath string) error {
	// Directly use LoadImage function to avoid dangling pointer issues
	inputSDImage := sd.GenerateImageFromPath(inputImagePath)
	fmt.Printf("inputSDImage: %+v", inputSDImage)
	outputSDImage := us.ctx.Upscale(inputSDImage, upscaleFactor)
	fmt.Printf("outputSDImage: %+v", outputSDImage)

	defer us.ctx.Free()

	// Save image
	err := sd.SaveImage(&outputSDImage, outputImagePath)
	if err != nil {
		return fmt.Errorf("failed to save image: %v", err)
	}

	return nil
}

// Convert model conversion function, convert a model to gguf format.
// inputPath: Path to the input model.
// vaePath: Path to the vae.
// outputPath: Path to save the converted model.
// outputType: The weight type (default: auto).
// tensorTypeRules: Weight type per tensor pattern (example: "^vae\\\\.=f16,model\\\\.=q8_0")
func Convert(inputPath, vaePath, outputPath, outputType, tensorTypeRules string, convertName bool) error {
	if outputPath == "" {
		outputPath = "output.gguf"
	}

	if outputType == "" {
		outputType = "default"
	}

	var outputSDType sd.SDType
	if value, ok := SDTypeMap[outputType]; ok {
		outputSDType = value
	} else {
		return fmt.Errorf("Invalid SDType: %s", outputType)
	}

	res := sd.Convert(inputPath, vaePath, outputPath, outputSDType, tensorTypeRules, convertName)
	if !res {
		return errors.New("failed to convert model")
	}
	return nil
}
