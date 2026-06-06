package sd

import "unsafe"

// cFree holds the C runtime free(3) function once bound by Load (best-effort,
// via bindCFree). When it is nil — because the native library has not been
// loaded yet or the C runtime could not be resolved — FreeImage and FreeImages
// are no-ops, so they are always safe to call.
var cFree func(unsafe.Pointer)

// FreeImage frees a single SDImage returned by the native library: its pixel
// buffer and the struct allocation itself. It is safe to call with a nil image
// or when the free binding is unavailable (in which case it does nothing).
func FreeImage(img *SDImage) {
	FreeImages(img, 1)
}

// FreeImages frees an array of count SDImages returned by the native library
// (for example the result of generate_image, which mallocs an array of
// BatchCount sd_image_t, each owning a malloc'd pixel buffer). It frees every
// image's pixel buffer and then the backing array allocation. It is safe to
// call with a nil pointer, a non-positive count, or when the free binding is
// unavailable (in which case it does nothing).
func FreeImages(imgs *SDImage, count int) {
	if cFree == nil || imgs == nil || count <= 0 {
		return
	}

	base := unsafe.Pointer(imgs)
	stride := unsafe.Sizeof(SDImage{})
	for i := 0; i < count; i++ {
		img := (*SDImage)(unsafe.Add(base, uintptr(i)*stride))
		if img.Data != nil {
			cFree(unsafe.Pointer(img.Data))
			img.Data = nil
		}
	}
	cFree(base)
}
