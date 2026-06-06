//go:build darwin || linux

package sd

import (
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// libcCandidates lists the C runtime shared libraries to try when binding the
// free(3) symbol. On Unix the sd.cpp output buffers are allocated with the
// system malloc, so the system free is the correct deallocator.
func libcCandidates() []string {
	if runtime.GOOS == "darwin" {
		return []string{"libSystem.B.dylib", "libc.dylib"}
	}
	return []string{"libc.so.6", "libc.so"}
}

// bindCFree resolves the C runtime free(3) function into cFree. It is
// best-effort and idempotent: on any failure cFree is left nil (FreeImage /
// FreeImages then become no-ops) and it never panics. Called from Load while
// loadMu is held.
func bindCFree() {
	if cFree != nil {
		return
	}

	// purego.RegisterLibFunc panics if the symbol is missing; never let that
	// escape — a missing free binding must degrade gracefully, not crash.
	defer func() { _ = recover() }()

	for _, name := range libcCandidates() {
		handle, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil || handle == 0 {
			continue
		}
		var fn func(unsafe.Pointer)
		purego.RegisterLibFunc(&fn, handle, "free")
		cFree = fn
		return
	}
}
