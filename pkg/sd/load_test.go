package sd

import (
	"path/filepath"
	"testing"
)

// withGPUStubs swaps the host-probing GPU/Vulkan detectors for fixed values for
// the duration of a test, restoring them on cleanup.
func withGPUStubs(t *testing.T, vendor string, vulkan bool) {
	t.Helper()
	origVendor, origVulkan := detectGPUVendor, vulkanLoaderPresent
	detectGPUVendor = func() (string, error) { return vendor, nil }
	vulkanLoaderPresent = func() bool { return vulkan }
	t.Cleanup(func() { detectGPUVendor, vulkanLoaderPresent = origVendor, origVulkan })
}

// hasVariant reports whether candidates contains libDir/<variant>/name.
func hasVariant(candidates []string, libDir, variant, name string) bool {
	want := filepath.Join(libDir, variant, name)
	for _, c := range candidates {
		if c == want {
			return true
		}
	}
	return false
}

// indexOfVariant returns the position of libDir/<variant>/name, or -1.
func indexOfVariant(candidates []string, libDir, variant, name string) int {
	want := filepath.Join(libDir, variant, name)
	for i, c := range candidates {
		if c == want {
			return i
		}
	}
	return -1
}

const (
	tNVIDIA = "NVIDIA"
	tAMD    = "AMD"
)

// TestWindowsLibCandidates_NvidiaPrefersVulkan is the #996-followup regression:
// on an NVIDIA host (no cuda13 build bundled on Windows) the Vulkan GPU build
// must be tried before the CPU fallback, so image generation runs on the GPU
// instead of silently on CPU.
func TestWindowsLibCandidates_NvidiaPrefersVulkan(t *testing.T) {
	t.Setenv("SD_VK_DEVICE", "")
	withGPUStubs(t, tNVIDIA, true)
	const libDir, name = "libs", "stable-diffusion.dll"

	got := windowsLibCandidates(libDir, name)

	vk := indexOfVariant(got, libDir, "vulkan", name)
	cpu := indexOfVariant(got, libDir, GetCpuAVX(), name)
	if vk < 0 {
		t.Fatalf("vulkan variant not offered for an NVIDIA host: %v", got)
	}
	if cpu < 0 || vk > cpu {
		t.Fatalf("vulkan must precede the CPU fallback; got %v", got)
	}
	if !hasVariant(got, libDir, "cuda13", name) {
		t.Fatalf("vendor-optimal cuda13 should still be offered first: %v", got)
	}
	if got[len(got)-1] != filepath.Join(libDir, GetCpuAVX(), name) {
		t.Fatalf("CPU build must remain the last fallback; got %v", got)
	}
}

// TestWindowsLibCandidates_AmdPrefersVulkan mirrors the NVIDIA case for AMD
// (rocm vendor build first, then vulkan, then CPU).
func TestWindowsLibCandidates_AmdPrefersVulkan(t *testing.T) {
	t.Setenv("SD_VK_DEVICE", "")
	withGPUStubs(t, tAMD, true)
	const libDir, name = "libs", "stable-diffusion.dll"

	got := windowsLibCandidates(libDir, name)
	if !hasVariant(got, libDir, "rocm", name) || !hasVariant(got, libDir, "vulkan", name) {
		t.Fatalf("expected rocm + vulkan candidates for AMD; got %v", got)
	}
}

// TestWindowsLibCandidates_NoVulkanLoaderNoVulkan ensures we don't offer the
// Vulkan build when no Vulkan loader is installed — it would just fail to load.
func TestWindowsLibCandidates_NoVulkanLoaderNoVulkan(t *testing.T) {
	t.Setenv("SD_VK_DEVICE", "")
	withGPUStubs(t, tNVIDIA, false) // GPU present, but vulkan-1.dll absent
	const libDir, name = "libs", "stable-diffusion.dll"

	got := windowsLibCandidates(libDir, name)
	if hasVariant(got, libDir, "vulkan", name) {
		t.Fatalf("vulkan offered without a Vulkan loader present; got %v", got)
	}
}

// TestWindowsLibCandidates_NoGPUCpuOnly verifies a host with no detectable GPU
// falls straight to the CPU build.
func TestWindowsLibCandidates_NoGPUCpuOnly(t *testing.T) {
	t.Setenv("SD_VK_DEVICE", "")
	withGPUStubs(t, "", false)
	const libDir, name = "libs", "stable-diffusion.dll"

	got := windowsLibCandidates(libDir, name)
	if len(got) != 1 || got[0] != filepath.Join(libDir, GetCpuAVX(), name) {
		t.Fatalf("expected CPU-only candidate, got %v", got)
	}
}

// TestWindowsLibCandidates_SDVKDeviceForcesVulkan verifies the explicit override
// adds the Vulkan build even with no GPU detected and no loader probe — honoring
// an operator who opted in.
func TestWindowsLibCandidates_SDVKDeviceForcesVulkan(t *testing.T) {
	t.Setenv("SD_VK_DEVICE", "true")
	withGPUStubs(t, "", false)
	const libDir, name = "libs", "stable-diffusion.dll"

	got := windowsLibCandidates(libDir, name)
	if !hasVariant(got, libDir, "vulkan", name) {
		t.Fatalf("SD_VK_DEVICE=true should force the vulkan candidate; got %v", got)
	}
}

// TestLoadMissingLibReturnsError verifies that loading from a directory with no
// shared library returns an error instead of panicking, so importers stay
// healthy when the native lib is absent.
func TestLoadMissingLibReturnsError(t *testing.T) {
	// The premise only holds while the library hasn't already been loaded in
	// this process; if it has, Load is a no-op and returns nil by design.
	if loaded {
		t.Skip("library already loaded in this process; missing-lib path not exercisable")
	}
	if err := Load(t.TempDir()); err == nil {
		t.Fatal("expected an error loading from an empty dir, got nil")
	}
	if loaded {
		t.Fatal("loaded flag should remain false after a failed Load")
	}
}

// TestLibCandidatesEmptyDir verifies an empty libDir resolves to the bare
// filename so the OS default search path is used.
func TestLibCandidatesEmptyDir(t *testing.T) {
	got := libCandidates("")
	if len(got) != 1 || got[0] != libFileName() {
		t.Fatalf("expected [%q], got %v", libFileName(), got)
	}
}
