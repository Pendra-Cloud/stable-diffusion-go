package sd

import (
	"runtime"
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
// uintptr-sized value; a void Go callback panics. The log/progress/preview
// callback types (and their wrapper closures in SetLogCallback /
// SetProgressCallback / SetPreviewCallback) therefore return uintptr.
//
// The log callback is the one the in-process worker actually registers (on the
// first generate), so it is the production-critical path and must wrap on every
// platform — including Windows, where the bug bit. The preview callback has no
// float in its signature and must also wrap everywhere.
//
// No native library is needed: the panic (if any) happens at callback
// compilation, not at the C call.
func TestCallbackSignaturesAcceptedByPurego(t *testing.T) {
	mustWrap := func(t *testing.T, name string, fn any) {
		t.Helper()
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("purego.NewCallback(%s) panicked (void-callback regression): %v", name, r)
			}
		}()
		if got := purego.NewCallback(fn); got == 0 {
			t.Fatalf("purego.NewCallback(%s) returned a 0 handle", name)
		}
	}

	var logCB SDLogCallback = func(level SDLogLevel, text *uint8, data unsafe.Pointer) uintptr {
		return 0
	}
	var previewCB SDPreviewCallback = func(step, frameCount int32, frames *SDImage, isNoisy bool, data unsafe.Pointer) uintptr {
		return 0
	}

	mustWrap(t, "SDLogCallback", logCB)
	mustWrap(t, "SDPreviewCallback", previewCB)
}

// TestProgressCallbackUnsupportedOnWindows documents and locks in a SEPARATE,
// pre-existing limitation surfaced by the work above: SDProgressCallback takes
// a `float32` argument, and Windows' syscall.NewCallback (which purego uses)
// supports neither float arguments nor float returns. So the progress callback
// can never be wrapped on Windows, regardless of its return type — it panics
// with "float arguments not supported". This is independent of the
// image-generation fix (the worker registers only the log callback) and would
// require dropping the float from the C-ABI signature to lift. We assert the
// status quo so a future change to SetProgressCallback's Windows behaviour is a
// deliberate, visible decision.
func TestProgressCallbackUnsupportedOnWindows(t *testing.T) {
	var progressCB SDProgressCallback = func(step, steps int32, tm float32, data unsafe.Pointer) uintptr {
		return 0
	}

	if runtime.GOOS == "windows" {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected purego.NewCallback(SDProgressCallback) to panic on Windows (float arg); it did not — update this test if the limitation was lifted")
			}
		}()
		_ = purego.NewCallback(progressCB)
		return
	}

	// Everywhere else the float arg is fine and the uintptr return is accepted.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("purego.NewCallback(SDProgressCallback) panicked off-Windows: %v", r)
		}
	}()
	if got := purego.NewCallback(progressCB); got == 0 {
		t.Fatal("purego.NewCallback(SDProgressCallback) returned a 0 handle")
	}
}
