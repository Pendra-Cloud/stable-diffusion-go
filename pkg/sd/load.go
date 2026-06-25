package sd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/ebitengine/purego"
)

// Dynamic library handle, populated by Load.
var libSD uintptr

// loadMu guards the lazy, idempotent Load.
var (
	loadMu sync.Mutex
	loaded bool
)

// libFileName returns the platform-specific shared library filename.
func libFileName() string {
	switch runtime.GOOS {
	case "windows":
		return "stable-diffusion.dll"
	case "darwin":
		return "libstable-diffusion.dylib"
	default: // linux and other unix
		return "libstable-diffusion.so"
	}
}

// libCandidates returns the ordered list of paths to try when opening the
// library from libDir. An empty libDir yields a single bare-filename candidate
// so the OS dynamic-loader search path is used.
func libCandidates(libDir string) []string {
	name := libFileName()
	if libDir == "" {
		return []string{name}
	}

	if runtime.GOOS == "windows" {
		return windowsLibCandidates(libDir, name)
	}
	return []string{filepath.Join(libDir, name)}
}

// windowsLibCandidates selects GPU-accelerated variant subdirectories within
// libDir based on the detected GPU, always falling back to the CPU (AVX)
// variant. The selection logic operates entirely within libDir.
func windowsLibCandidates(libDir, name string) []string {
	var candidates []string

	if strings.ToLower(os.Getenv("SD_VK_DEVICE")) == "true" {
		if gpu, err := GetVulkanGPU(); err == nil && gpu != "" {
			candidates = append(candidates, filepath.Join(libDir, "vulkan", name))
		}
	} else if gpu, err := GetGPUName(); err == nil {
		switch gpu {
		case "NVIDIA":
			candidates = append(candidates, filepath.Join(libDir, "cuda13", name))
		case "AMD":
			candidates = append(candidates, filepath.Join(libDir, "rocm", name))
		}
	}

	// CPU fallback (avx512/avx2/avx/noavx).
	candidates = append(candidates, filepath.Join(libDir, GetCpuAVX(), name))
	return candidates
}

// Load resolves and dlopens the stable-diffusion shared library from libDir.
// An empty libDir falls back to the OS default search path.
// Load is idempotent and safe for concurrent use; it returns an error (and
// never panics) when the library is absent or incompatible, so that merely
// importing this package — or calling Load when the library is missing — keeps
// the caller healthy.
func Load(libDir string) (err error) {
	loadMu.Lock()
	defer loadMu.Unlock()

	if loaded {
		return nil
	}

	// purego.RegisterLibFunc panics when a symbol is missing; convert any panic
	// across the FFI boundary into an error so the caller never crashes. Close
	// the handle we opened so it isn't leaked (and the DLL isn't left locked on
	// Windows) when symbol registration fails partway through.
	defer func() {
		if r := recover(); r != nil {
			if libSD != 0 {
				_ = closeLibrary(libSD)
			}
			libSD = 0
			err = fmt.Errorf("stable-diffusion: failed to load library: %v", r)
		}
	}()

	candidates := libCandidates(libDir)

	var (
		handle  uintptr
		lastErr error
	)
	for _, path := range candidates {
		handle, lastErr = openLibrary(path)
		if lastErr == nil && handle != 0 {
			break
		}
	}
	if handle == 0 {
		if lastErr == nil {
			lastErr = fmt.Errorf("no library candidates for GOOS %q", runtime.GOOS)
		}
		return fmt.Errorf("stable-diffusion: failed to load library from %q: %w", libDir, lastErr)
	}

	libSD = handle
	registerFunctions()
	bindCFree() // best-effort; FreeImage/FreeImages no-op if this fails

	loaded = true
	return nil
}

// registerFunctions binds the purego symbols from the loaded library. It panics
// (via purego.RegisterLibFunc) if a symbol is missing; Load recovers from that.
func registerFunctions() {
	purego.RegisterLibFunc(&sdSetLogCallback, libSD, "sd_set_log_callback")
	purego.RegisterLibFunc(&sdSetProgressCallback, libSD, "sd_set_progress_callback")
	purego.RegisterLibFunc(&sdSetPreviewCallback, libSD, "sd_set_preview_callback")
	purego.RegisterLibFunc(&sdGetNumPhysicalCores, libSD, "sd_get_num_physical_cores")
	purego.RegisterLibFunc(&sdGetSystemInfo, libSD, "sd_get_system_info")
	purego.RegisterLibFunc(&sdTypeName, libSD, "sd_type_name")
	purego.RegisterLibFunc(&strToSDType, libSD, "str_to_sd_type")
	purego.RegisterLibFunc(&sdRngTypeName, libSD, "sd_rng_type_name")
	purego.RegisterLibFunc(&strToRngType, libSD, "str_to_rng_type")
	purego.RegisterLibFunc(&sdSampleMethodName, libSD, "sd_sample_method_name")
	purego.RegisterLibFunc(&strToSampleMethod, libSD, "str_to_sample_method")
	purego.RegisterLibFunc(&sdSchedulerName, libSD, "sd_scheduler_name")
	purego.RegisterLibFunc(&strToScheduler, libSD, "str_to_scheduler")
	purego.RegisterLibFunc(&sdPredictionName, libSD, "sd_prediction_name")
	purego.RegisterLibFunc(&strToPrediction, libSD, "str_to_prediction")
	purego.RegisterLibFunc(&sdPreviewName, libSD, "sd_preview_name")
	purego.RegisterLibFunc(&strToPreview, libSD, "str_to_preview")
	purego.RegisterLibFunc(&sdLoraApplyModeName, libSD, "sd_lora_apply_mode_name")
	purego.RegisterLibFunc(&strToLoraApplyMode, libSD, "str_to_lora_apply_mode")
	purego.RegisterLibFunc(&sdCacheParamsInit, libSD, "sd_cache_params_init")
	purego.RegisterLibFunc(&sdContextParamsInit, libSD, "sd_ctx_params_init")
	purego.RegisterLibFunc(&sdContextParamsToStr, libSD, "sd_ctx_params_to_str")
	purego.RegisterLibFunc(&newSDContext, libSD, "new_sd_ctx")
	purego.RegisterLibFunc(&freeSDContext, libSD, "free_sd_ctx")
	purego.RegisterLibFunc(&sdSampleParamsInit, libSD, "sd_sample_params_init")
	purego.RegisterLibFunc(&sdSampleParamsToStr, libSD, "sd_sample_params_to_str")
	purego.RegisterLibFunc(&sdGetDefaultSampleMethod, libSD, "sd_get_default_sample_method")
	purego.RegisterLibFunc(&sdGetDefaultScheduler, libSD, "sd_get_default_scheduler")
	purego.RegisterLibFunc(&sdImgGenParamsInit, libSD, "sd_img_gen_params_init")
	purego.RegisterLibFunc(&sdImgGenParamsToStr, libSD, "sd_img_gen_params_to_str")
	purego.RegisterLibFunc(&generateImage, libSD, "generate_image")
	purego.RegisterLibFunc(&sdVidGenParamsInit, libSD, "sd_vid_gen_params_init")
	purego.RegisterLibFunc(&generateVideo, libSD, "generate_video")
	purego.RegisterLibFunc(&newUpscalerContext, libSD, "new_upscaler_ctx")
	purego.RegisterLibFunc(&freeUpscalerContext, libSD, "free_upscaler_ctx")
	purego.RegisterLibFunc(&upscale, libSD, "upscale")
	purego.RegisterLibFunc(&getUpscaleFactor, libSD, "get_upscale_factor")
	purego.RegisterLibFunc(&convert, libSD, "convert")
	purego.RegisterLibFunc(&preprocessCanny, libSD, "preprocess_canny")
	purego.RegisterLibFunc(&sdCommit, libSD, "sd_commit")
	purego.RegisterLibFunc(&sdVersion, libSD, "sd_version")
	purego.RegisterLibFunc(&sdCtxSupportsImageGeneration, libSD, "sd_ctx_supports_image_generation")
	purego.RegisterLibFunc(&sdCtxSupportsVideoGeneration, libSD, "sd_ctx_supports_video_generation")
	purego.RegisterLibFunc(&sdHiresUpscalerName, libSD, "sd_hires_upscaler_name")
	purego.RegisterLibFunc(&strToSDHiresUpscaler, libSD, "str_to_sd_hires_upscaler")
	purego.RegisterLibFunc(&sdHiresParamsInit, libSD, "sd_hires_params_init")
	purego.RegisterLibFunc(&freeSDAudio, libSD, "free_sd_audio")
}
