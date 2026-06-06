package sd

import (
	"bytes"
	"image/png"
	"testing"
)

// newTestImage builds an SDImage backed by a Go-allocated 3-channel RGB buffer.
// The returned slice must be kept alive by the caller for as long as the image
// is used (Data points into it).
func newTestImage(w, h uint32, fill func(x, y uint32) (r, g, b uint8)) (*SDImage, []uint8) {
	data := make([]uint8, int(w*h*3))
	for y := uint32(0); y < h; y++ {
		for x := uint32(0); x < w; x++ {
			i := (y*w + x) * 3
			r, g, b := fill(x, y)
			data[i], data[i+1], data[i+2] = r, g, b
		}
	}
	return &SDImage{Width: w, Height: h, Channel: 3, Data: &data[0]}, data
}

func TestEncodePNGRoundTrip(t *testing.T) {
	const w, h = 4, 3
	img, data := newTestImage(w, h, func(x, y uint32) (uint8, uint8, uint8) {
		// Distinct, position-dependent values so we can verify pixels survive.
		return uint8(x * 10), uint8(y * 10), uint8(x + y)
	})
	defer func() { _ = data }() // keep buffer alive until after encode

	encoded, err := EncodePNG(img)
	if err != nil {
		t.Fatalf("EncodePNG returned error: %v", err)
	}
	if len(encoded) == 0 {
		t.Fatal("EncodePNG returned no bytes")
	}

	decoded, err := png.Decode(bytes.NewReader(encoded))
	if err != nil {
		t.Fatalf("failed to decode PNG produced by EncodePNG: %v", err)
	}

	b := decoded.Bounds()
	if b.Dx() != w || b.Dy() != h {
		t.Fatalf("decoded dimensions = %dx%d, want %dx%d", b.Dx(), b.Dy(), w, h)
	}

	// Sample a pixel and confirm RGB survived the round-trip (alpha is opaque).
	const sx, sy = 2, 1
	r, g, bl, a := decoded.At(sx, sy).RGBA()
	wantR, wantG, wantB := uint8(sx*10), uint8(sy*10), uint8(sx+sy)
	if uint8(r>>8) != wantR || uint8(g>>8) != wantG || uint8(bl>>8) != wantB || uint8(a>>8) != 255 {
		t.Fatalf("pixel (%d,%d) = (%d,%d,%d,%d), want (%d,%d,%d,255)",
			sx, sy, uint8(r>>8), uint8(g>>8), uint8(bl>>8), uint8(a>>8), wantR, wantG, wantB)
	}
}

func TestEncodePNGNilReturnsError(t *testing.T) {
	if _, err := EncodePNG(nil); err == nil {
		t.Fatal("EncodePNG(nil) = nil error, want error")
	}
	if _, err := EncodePNG(&SDImage{Width: 1, Height: 1, Channel: 3}); err == nil {
		t.Fatal("EncodePNG with nil Data = nil error, want error")
	}
}
