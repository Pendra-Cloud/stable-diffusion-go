package sd

import "testing"

// When the native library has not been loaded, cFree is nil and the free
// helpers must be safe no-ops — never panicking and never calling into a nil
// deallocator. This covers the graceful-degradation path (e.g. unit tests and
// callers that never load a lib).
func TestFreeImageNoBindingIsSafe(t *testing.T) {
	if cFree != nil {
		t.Skip("cFree is bound; this test covers the unbound no-op path")
	}

	// Nil and zero inputs.
	FreeImage(nil)
	FreeImages(nil, 0)
	FreeImages(nil, 5)

	// Non-nil, Go-allocated image: with no binding this must not attempt a free
	// and must leave the data pointer untouched.
	data := make([]uint8, 3)
	img := &SDImage{Width: 1, Height: 1, Channel: 3, Data: &data[0]}
	FreeImage(img)
	if img.Data == nil {
		t.Fatal("FreeImage cleared Data without a free binding; want no-op")
	}
}
