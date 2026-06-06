package sd

import (
	"bytes"
	"fmt"
	"golang.org/x/sys/cpu"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"
)

// GetVulkanGPU gets Vulkan GPU information
func GetVulkanGPU() (string, error) {
	// Try to detect Vulkan device using vulkaninfo command
	cmd := exec.Command("vulkaninfo", "--summary")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("vulkaninfo not available or failed: %w", err)
	}

	outputStr := string(output)

	// Check if there's an NVIDIA device
	if strings.Contains(strings.ToUpper(outputStr), "NVIDIA") {
		return "NVIDIA (Vulkan)", nil
	}

	// Check if there's an AMD device
	if strings.Contains(strings.ToUpper(outputStr), "AMD") ||
		strings.Contains(strings.ToUpper(outputStr), "RADEON") {
		return "AMD (Vulkan)", nil
	}

	// Check if there's an Intel device
	if strings.Contains(strings.ToUpper(outputStr), "INTEL") {
		return "Intel (Vulkan)", nil
	}

	return "Vulkan Device", nil
}

// GetGPUName gets GPU name
func GetGPUName() (string, error) {
	cmd := exec.Command("wmic", "path", "win32_VideoController", "get", "name")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	lines := strings.Split(string(output), "\n")
	var gpuName string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" && trimmed != "Name" {
			if strings.Contains(strings.ToUpper(trimmed), "NVIDIA") {
				gpuName = "NVIDIA"
				break
			} else if strings.Contains(strings.ToUpper(trimmed), "AMD") {
				gpuName = "AMD"
				break
			} else if strings.Contains(strings.ToUpper(trimmed), "INTEL") {
				gpuName = "Intel"
				break
			}
		}
	}

	return gpuName, nil
}

// GetCpuAVX gets CPU AVX instruction set
func GetCpuAVX() string {
	if cpu.X86.HasAVX512 {
		return "avx512"
	} else if cpu.X86.HasAVX2 {
		return "avx2"
	} else if cpu.X86.HasAVX {
		return "avx"
	} else {
		return "noavx"
	}
}

// toRGBA converts a raw SDImage into a Go *image.RGBA. It reads the first three
// channels as RGB (ignoring any extra channels) and adds an opaque alpha
// channel. Malformed or oversized inputs return an error rather than panicking.
func toRGBA(img *SDImage) (*image.RGBA, error) {
	if img == nil || img.Data == nil {
		return nil, fmt.Errorf("invalid image data")
	}
	if img.Channel < 3 {
		return nil, fmt.Errorf("unsupported channel count %d: need at least 3 (RGB)", img.Channel)
	}
	if img.Width == 0 || img.Height == 0 {
		return nil, fmt.Errorf("invalid image dimensions %dx%d", img.Width, img.Height)
	}

	// Compute the pixel-buffer length in uint64 to avoid the uint32 overflow
	// that a direct Width*Height*Channel multiply would risk, then guard it
	// against the platform int size before slicing/indexing.
	count := uint64(img.Width) * uint64(img.Height) * uint64(img.Channel)
	if count > uint64(math.MaxInt) {
		return nil, fmt.Errorf("image too large: %d bytes", count)
	}

	// Create RGBA image
	bounds := image.Rect(0, 0, int(img.Width), int(img.Height))
	rgba := image.NewRGBA(bounds)

	// Convert raw data to RGBA format, reading 3 channels per pixel.
	data := unsafe.Slice(img.Data, int(count))
	channel := int(img.Channel)
	width := int(img.Width)
	pixels := width * int(img.Height)
	for i := 0; i < pixels; i++ {
		index := i * channel
		x := i % width
		y := i / width

		rgba.Set(x, y, color.RGBA{data[index], data[index+1], data[index+2], 255}) // opaque alpha
	}

	return rgba, nil
}

// EncodePNG encodes an SDImage as PNG and returns the bytes, without touching
// the filesystem — useful for serving generated images directly (e.g. base64
// over an API) instead of round-tripping through a file.
func EncodePNG(img *SDImage) ([]byte, error) {
	rgba, err := toRGBA(img)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, rgba); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SaveImage saves SDImage as PNG file
func SaveImage(img *SDImage, path string) error {
	rgba, err := toRGBA(img)
	if err != nil {
		return err
	}

	// Create output file
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Save as PNG
	return png.Encode(file, rgba)
}

// LoadImage loads image from file and converts to SDImage format
func LoadImage(path string) (SDImage, error) {
	// Open image file
	file, err := os.Open(path)
	if err != nil {
		return SDImage{}, fmt.Errorf("failed to open image file: %v", err)
	}
	defer file.Close()

	// Decode image
	img, _, err := image.Decode(file)
	if err != nil {
		return SDImage{}, fmt.Errorf("failed to decode image: %v", err)
	}

	// Get image bounds
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Calculate RGB data size
	channel := 3 // RGB format
	dataSize := width * height * channel

	// Create data array
	data := make([]uint8, dataSize)

	// Convert image data to RGB format
	index := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			// Get pixel color (ignore Alpha channel)
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert RGBA to 8-bit RGB
			data[index] = uint8(r >> 8) // R
			index++
			data[index] = uint8(g >> 8) // G
			index++
			data[index] = uint8(b >> 8) // B
			index++
		}
	}

	// Create sd.SDImage struct
	result := SDImage{
		Width:   uint32(width),
		Height:  uint32(height),
		Channel: uint32(channel),
		Data:    &data[0], // pointer to first element of data array
	}

	return result, nil
}

// GenerateImageFromPath generates SDImage from path
func GenerateImageFromPath(imagePath string) SDImage {
	if imagePath == "" {
		return SDImage{}
	}

	img, err := LoadImage(imagePath)
	if err != nil {
		fmt.Println("Error loading image:", err)
		return SDImage{}
	}
	return img
}

func GenerateImagesFromPaths(path []string) *SDImage {
	if path == nil || len(path) == 0 {
		return nil
	}

	// Create SDImage slice
	images := make([]SDImage, 0, len(path))

	// Iterate through all paths, generate SDImage
	for _, p := range path {
		if p == "" {
			continue
		}

		img := GenerateImageFromPath(p)
		// Only add valid images
		if img.Data != nil {
			images = append(images, img)
		}
	}

	if len(images) == 0 {
		return nil
	}

	// Return pointer to first element, so all elements can be accessed via pointer offset
	return &images[0]
}

// SaveFrames saves all video frames as PNG files
func SaveFrames(frames []SDImage, outputDir string) error {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	// Save each frame
	for i, frame := range frames {
		framePath := filepath.Join(outputDir, fmt.Sprintf("frame_%04d.png", i+1))
		if err := SaveImage(&frame, framePath); err != nil {
			return fmt.Errorf("failed to save frame %d: %v", i+1, err)
		}
	}

	return nil
}

// EncodeVideo encodes PNG frame sequence to video using FFmpeg
func EncodeVideo(inputDir, outputPath string, framerate int) error {
	// Check if FFmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("ffmpeg not found: %v", err)
	}

	// Build FFmpeg command
	cmd := exec.Command(
		"ffmpeg",
		"-y", // overwrite output file
		"-framerate", strconv.Itoa(framerate),
		"-i", filepath.Join(inputDir, "frame_%04d.png"),
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		outputPath,
	)

	// Set output
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Execute command
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg failed: %v", err)
	}

	return nil
}

// CleanupTempDir cleans up temporary directory
func CleanupTempDir(tempDir string) error {
	return os.RemoveAll(tempDir)
}
