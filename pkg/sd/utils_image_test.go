package sd

import (
	"os"
	"path/filepath"
	"testing"
)

// TestToRGBAReadsFirstThreeChannels feeds a 4-channel (RGBA) buffer and checks
// toRGBA reads R,G,B from the first three channels, skips the 4th, and forces an
// opaque alpha — i.e. the per-pixel stride honours Channel, not a hard-coded 3.
func TestToRGBAReadsFirstThreeChannels(t *testing.T) {
	const w, h = 2, 1
	// Pixel0 = (10,20,30, 40-ignored), Pixel1 = (50,60,70, 80-ignored)
	data := []uint8{10, 20, 30, 40, 50, 60, 70, 80}
	img := &SDImage{Width: w, Height: h, Channel: 4, Data: &data[0]}

	rgba, err := toRGBA(img)
	if err != nil {
		t.Fatalf("toRGBA returned error: %v", err)
	}
	for x, want := range [][3]uint8{{10, 20, 30}, {50, 60, 70}} {
		r, g, b, a := rgba.At(x, 0).RGBA()
		if uint8(r>>8) != want[0] || uint8(g>>8) != want[1] || uint8(b>>8) != want[2] {
			t.Errorf("pixel %d RGB = (%d,%d,%d), want %v", x, uint8(r>>8), uint8(g>>8), uint8(b>>8), want)
		}
		if uint8(a>>8) != 255 {
			t.Errorf("pixel %d alpha = %d, want 255 (opaque)", x, uint8(a>>8))
		}
	}
}

// TestSaveLoadImageRoundTrip writes an SDImage to a PNG via SaveImage and reads
// it back via LoadImage, confirming dimensions, channel count, and pixel values
// survive the lossless round trip.
func TestSaveLoadImageRoundTrip(t *testing.T) {
	const w, h = 4, 3
	src, data := newTestImage(w, h, func(x, y uint32) (uint8, uint8, uint8) {
		return uint8(x * 10), uint8(y * 20), uint8(x + y + 1)
	})
	defer func() { _ = data }()

	path := filepath.Join(t.TempDir(), "round.png")
	if err := SaveImage(src, path); err != nil {
		t.Fatalf("SaveImage: %v", err)
	}

	loaded, err := LoadImage(path)
	if err != nil {
		t.Fatalf("LoadImage: %v", err)
	}
	if loaded.Width != w || loaded.Height != h {
		t.Fatalf("dimensions = %dx%d, want %dx%d", loaded.Width, loaded.Height, w, h)
	}
	if loaded.Channel != 3 {
		t.Fatalf("channel = %d, want 3", loaded.Channel)
	}
	if loaded.Data == nil {
		t.Fatal("loaded image has nil data")
	}

	// Spot-check pixel (2,1) survived the round trip.
	rgba, err := toRGBA(&loaded)
	if err != nil {
		t.Fatalf("toRGBA on loaded image: %v", err)
	}
	r, g, b, _ := rgba.At(2, 1).RGBA()
	got := [3]uint8{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
	want := [3]uint8{2 * 10, 1 * 20, 2 + 1 + 1}
	if got != want {
		t.Fatalf("pixel (2,1) = %v, want %v", got, want)
	}
}

// TestLoadImageMissingFileErrors confirms a missing path is an error, not a panic.
func TestLoadImageMissingFileErrors(t *testing.T) {
	if _, err := LoadImage(filepath.Join(t.TempDir(), "nope.png")); err == nil {
		t.Fatal("LoadImage(missing) = nil error, want error")
	}
}

// TestGenerateImageFromPath covers the empty-path and missing-file cases, both
// of which must yield a zero-value SDImage (nil Data) rather than panicking.
func TestGenerateImageFromPath(t *testing.T) {
	if img := GenerateImageFromPath(""); img.Data != nil {
		t.Error("GenerateImageFromPath(\"\") returned non-nil Data")
	}
	if img := GenerateImageFromPath(filepath.Join(t.TempDir(), "missing.png")); img.Data != nil {
		t.Error("GenerateImageFromPath(missing) returned non-nil Data")
	}
}

// TestGenerateImagesFromPaths covers nil/empty/all-invalid (=> nil) and the
// mixed case where at least one path is valid (=> non-nil).
func TestGenerateImagesFromPaths(t *testing.T) {
	if GenerateImagesFromPaths(nil) != nil {
		t.Error("GenerateImagesFromPaths(nil) != nil")
	}
	if GenerateImagesFromPaths([]string{}) != nil {
		t.Error("GenerateImagesFromPaths(empty) != nil")
	}
	if GenerateImagesFromPaths([]string{"", filepath.Join(t.TempDir(), "x.png")}) != nil {
		t.Error("GenerateImagesFromPaths(all invalid) != nil")
	}

	// One valid image among invalid entries → non-nil result.
	src, data := newTestImage(2, 2, func(x, y uint32) (uint8, uint8, uint8) { return 1, 2, 3 })
	defer func() { _ = data }()
	path := filepath.Join(t.TempDir(), "valid.png")
	if err := SaveImage(src, path); err != nil {
		t.Fatalf("SaveImage: %v", err)
	}
	if got := GenerateImagesFromPaths([]string{"", path}); got == nil {
		t.Error("GenerateImagesFromPaths with one valid path returned nil")
	}
}

// TestSaveFrames writes a frame sequence and checks the expected 1-based,
// zero-padded filenames land in the output directory.
func TestSaveFrames(t *testing.T) {
	f1, d1 := newTestImage(2, 2, func(x, y uint32) (uint8, uint8, uint8) { return 0, 0, 0 })
	f2, d2 := newTestImage(2, 2, func(x, y uint32) (uint8, uint8, uint8) { return 255, 255, 255 })
	defer func() { _, _ = d1, d2 }()

	dir := filepath.Join(t.TempDir(), "frames")
	if err := SaveFrames([]SDImage{*f1, *f2}, dir); err != nil {
		t.Fatalf("SaveFrames: %v", err)
	}
	for _, name := range []string{"frame_0001.png", "frame_0002.png"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("expected %s to exist: %v", name, err)
		}
	}
}

// TestGetCpuAVX confirms the detector returns one of the known variant subdir
// names the Windows loader resolves (avx512/avx2/avx/noavx).
func TestGetCpuAVX(t *testing.T) {
	got := GetCpuAVX()
	switch got {
	case "avx512", "avx2", "avx", "noavx":
	default:
		t.Fatalf("GetCpuAVX() = %q, want one of avx512/avx2/avx/noavx", got)
	}
}
