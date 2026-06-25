package sd

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// TestCallbackSignaturesAcceptedByPurego is the regression guard for the
// Windows image-generation crash:
//
//	stablediffusion generate: handler panic: compileCallback:
//	expected function with one uintptr-sized result
//
// purego turns a Go func passed as a C callback into a machine callback via
// purego.NewCallback, which on Windows delegates to syscall.NewCallback ->
// runtime.compileCallback. That path REQUIRES the callback return exactly one
// uintptr-sized value; a void Go callback panics. SDLogCallback /
// SDProgressCallback / SDPreviewCallback (and the wrapper closures in
// SetLogCallback / SetProgressCallback / SetPreviewCallback) therefore return
// uintptr. This test feeds each type through purego.NewCallback — the exact
// conversion the binding performs when it hands the closure to the C
// sd_set_*_callback functions — and fails if any signature regresses to void.
//
// It needs no native library: the panic happens at callback compilation, not
// at the C call. It runs on every platform (purego.NewCallback is portable);
// on Windows it exercises the failing syscall.NewCallback path directly.
func TestCallbackSignaturesAcceptedByPurego(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("purego.NewCallback panicked on a stable-diffusion callback "+
				"signature (Windows void-callback regression): %v", r)
		}
	}()

	var logCB SDLogCallback = func(level SDLogLevel, text *uint8, data unsafe.Pointer) uintptr {
		return 0
	}
	var progressCB SDProgressCallback = func(step, steps int32, t float32, data unsafe.Pointer) uintptr {
		return 0
	}
	var previewCB SDPreviewCallback = func(step, frameCount int32, frames *SDImage, isNoisy bool, data unsafe.Pointer) uintptr {
		return 0
	}

	// Each of these would panic with "expected function with one uintptr-sized
	// result" on Windows if the callback returned void.
	if got := purego.NewCallback(logCB); got == 0 {
		t.Fatal("NewCallback(SDLogCallback) returned 0 handle")
	}
	if got := purego.NewCallback(progressCB); got == 0 {
		t.Fatal("NewCallback(SDProgressCallback) returned 0 handle")
	}
	if got := purego.NewCallback(previewCB); got == 0 {
		t.Fatal("NewCallback(SDPreviewCallback) returned 0 handle")
	}
}
