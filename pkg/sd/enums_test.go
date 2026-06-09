package sd

import "testing"

// TestEnumABIValues pins the integer value of every exported enum constant.
//
// These are not arbitrary Go iota values — each must equal the integer of the
// matching C enum in stable-diffusion.h, because the binding passes them across
// the FFI boundary by value. Reordering a constant (or dropping one from a gap
// like the removed ggml quant types) would silently change behaviour at runtime
// without any compile error or symbol-gate failure. This table is the guard.
func TestEnumABIValues(t *testing.T) {
	cases := []struct {
		name string
		got  int32
		want int32
	}{
		// rng_type_t
		{"DefaultRNG", int32(DefaultRNG), 0},
		{"CUDARNG", int32(CUDARNG), 1},
		{"CPURNG", int32(CPURNG), 2},
		{"RNGTypeCount", int32(RNGTypeCount), 3},

		// sample_method_t
		{"EulerSampleMethod", int32(EulerSampleMethod), 0},
		{"EulerASampleMethod", int32(EulerASampleMethod), 1},
		{"HeunSampleMethod", int32(HeunSampleMethod), 2},
		{"DPM2SampleMethod", int32(DPM2SampleMethod), 3},
		{"DPMPP2SASampleMethod", int32(DPMPP2SASampleMethod), 4},
		{"DPMPP2MSampleMethod", int32(DPMPP2MSampleMethod), 5},
		{"DPMPP2Mv2SampleMethod", int32(DPMPP2Mv2SampleMethod), 6},
		{"IPNDMSampleMethod", int32(IPNDMSampleMethod), 7},
		{"IPNDMSampleMethodV", int32(IPNDMSampleMethodV), 8},
		{"LCMSampleMethod", int32(LCMSampleMethod), 9},
		{"DDIMTrailingSampleMethod", int32(DDIMTrailingSampleMethod), 10},
		{"TCDSampleMethod", int32(TCDSampleMethod), 11},
		{"ResMultistepSampleMethod", int32(ResMultistepSampleMethod), 12},
		{"Res2SSampleMethod", int32(Res2SSampleMethod), 13},
		{"ERSDESampleMethod", int32(ERSDESampleMethod), 14},
		{"EulerCFGPPSampleMethod", int32(EulerCFGPPSampleMethod), 15},
		{"EulerACFGPPSampleMethod", int32(EulerACFGPPSampleMethod), 16},
		{"EulerGESampleMethod", int32(EulerGESampleMethod), 17},
		{"SampleMethodCount", int32(SampleMethodCount), 18},

		// scheduler_t
		{"DiscreteScheduler", int32(DiscreteScheduler), 0},
		{"KarrasScheduler", int32(KarrasScheduler), 1},
		{"ExponentialScheduler", int32(ExponentialScheduler), 2},
		{"AYSScheduler", int32(AYSScheduler), 3},
		{"GITScheduler", int32(GITScheduler), 4},
		{"SGMUniformScheduler", int32(SGMUniformScheduler), 5},
		{"SimpleScheduler", int32(SimpleScheduler), 6},
		{"SmoothstepScheduler", int32(SmoothstepScheduler), 7},
		{"KLOptimalScheduler", int32(KLOptimalScheduler), 8},
		{"LCMScheduler", int32(LCMScheduler), 9},
		{"BongTangentScheduler", int32(BongTangentScheduler), 10},
		{"LTX2Scheduler", int32(LTX2Scheduler), 11},
		{"SchedulerCount", int32(SchedulerCount), 12},

		// prediction_t
		{"EPSPred", int32(EPSPred), 0},
		{"VPred", int32(VPred), 1},
		{"EDMVPred", int32(EDMVPred), 2},
		{"FlowPred", int32(FlowPred), 3},
		{"FluxFlowPred", int32(FluxFlowPred), 4},
		{"Flux2FlowPred", int32(Flux2FlowPred), 5},
		{"PredictionCount", int32(PredictionCount), 6},

		// sd_log_level_t
		{"SDLogDebug", int32(SDLogDebug), 0},
		{"SDLogInfo", int32(SDLogInfo), 1},
		{"SDLogWarn", int32(SDLogWarn), 2},
		{"SDLogError", int32(SDLogError), 3},

		// preview_t
		{"PreviewNone", int32(PreviewNone), 0},
		{"PreviewProj", int32(PreviewProj), 1},
		{"PreviewTAE", int32(PreviewTAE), 2},
		{"PreviewVAE", int32(PreviewVAE), 3},
		{"PreviewCount", int32(PreviewCount), 4},

		// lora_apply_mode_t
		{"LoraApplyAuto", int32(LoraApplyAuto), 0},
		{"LoraApplyImmediately", int32(LoraApplyImmediately), 1},
		{"LoraApplyAtRuntime", int32(LoraApplyAtRuntime), 2},
		{"LoraApplyModeCount", int32(LoraApplyModeCount), 3},

		// sd_cache_mode_t
		{"SDCacheDisabled", int32(SDCacheDisabled), 0},
		{"SDCacheEasycache", int32(SDCacheEasycache), 1},
		{"SDCacheUcache", int32(SDCacheUcache), 2},
		{"SDCacheDbcache", int32(SDCacheDbcache), 3},
		{"SDCacheTaylorseer", int32(SDCacheTaylorseer), 4},
		{"SDCacheCacheDit", int32(SDCacheCacheDit), 5},
		{"SDCacheSpectrum", int32(SDCacheSpectrum), 6},

		// sd_type_t — spot-check the explicitly numbered values, including the
		// ones after the removed-quant-type gaps where iota would be wrong.
		{"SDTypeF32", int32(SDTypeF32), 0},
		{"SDTypeF16", int32(SDTypeF16), 1},
		{"SDTypeQ4_0", int32(SDTypeQ4_0), 2},
		{"SDTypeQ4_1", int32(SDTypeQ4_1), 3},
		{"SDTypeQ5_0", int32(SDTypeQ5_0), 6},
		{"SDTypeQ8_0", int32(SDTypeQ8_0), 8},
		{"SDTypeQ4_K", int32(SDTypeQ4_K), 12},
		{"SDTypeBF16", int32(SDTypeBF16), 30},
		{"SDTypeTQ1_0", int32(SDTypeTQ1_0), 34},
		{"SDTypeMXFP4", int32(SDTypeMXFP4), 39},
		{"SDTypeNVFP4", int32(SDTypeNVFP4), 40},
		{"SDTypeQ1_0", int32(SDTypeQ1_0), 41},
		{"SDTypeCount", int32(SDTypeCount), 42},

		// sd_vae_format_t — note AUTO is -1.
		{"VAEFormatAuto", int32(VAEFormatAuto), -1},
		{"FluxVAEFormat", int32(FluxVAEFormat), 0},
		{"SD3VAEFormat", int32(SD3VAEFormat), 1},
		{"Flux2VAEFormat", int32(Flux2VAEFormat), 2},
		{"VAEFormatCount", int32(VAEFormatCount), 3},

		// sd_hires_upscaler_t
		{"HiresUpscalerNone", int32(HiresUpscalerNone), 0},
		{"HiresUpscalerLatent", int32(HiresUpscalerLatent), 1},
		{"HiresUpscalerLatentNearest", int32(HiresUpscalerLatentNearest), 2},
		{"HiresUpscalerLatentNearestExact", int32(HiresUpscalerLatentNearestExact), 3},
		{"HiresUpscalerLatentAntialiased", int32(HiresUpscalerLatentAntialiased), 4},
		{"HiresUpscalerLatentBicubic", int32(HiresUpscalerLatentBicubic), 5},
		{"HiresUpscalerLatentBicubicAntialiased", int32(HiresUpscalerLatentBicubicAntialiased), 6},
		{"HiresUpscalerLanczos", int32(HiresUpscalerLanczos), 7},
		{"HiresUpscalerNearest", int32(HiresUpscalerNearest), 8},
		{"HiresUpscalerModel", int32(HiresUpscalerModel), 9},
		{"HiresUpscalerCount", int32(HiresUpscalerCount), 10},
	}

	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d (C-ABI enum value drift)", c.name, c.got, c.want)
		}
	}
}
