package sd

import (
	"strings"
	"testing"
	"unsafe"
)

// TestCStringRoundTrip checks the Go<->C string helpers are inverses for a
// range of inputs, including multibyte UTF-8 (the helpers operate on raw bytes,
// so the byte sequence must survive intact).
func TestCStringRoundTrip(t *testing.T) {
	for _, s := range []string{
		"hello",
		"a/path/to/model.gguf",
		"with spaces and punctuation! ?",
		"unicode: café 🦊 日本語",
		strings.Repeat("x", 1000),
	} {
		c := CString(s)
		if c == nil {
			t.Errorf("CString(%q) = nil, want non-nil", s)
			continue
		}
		if got := CGoString(c); got != s {
			t.Errorf("round-trip CGoString(CString(%q)) = %q", s, got)
		}
	}
}

// TestCStringNullTerminated verifies CString writes a trailing NUL after the
// payload — the contract C callees rely on to find the end of the string.
func TestCStringNullTerminated(t *testing.T) {
	const s = "abc"
	c := CString(s)
	if c == nil {
		t.Fatal("CString returned nil for non-empty input")
	}
	buf := unsafe.Slice(c, len(s)+1)
	if got := string(buf[:len(s)]); got != s {
		t.Fatalf("payload = %q, want %q", got, s)
	}
	if buf[len(s)] != 0 {
		t.Fatalf("missing NUL terminator: byte[%d] = %d, want 0", len(s), buf[len(s)])
	}
}

// TestCStringEmptyIsNil documents the deliberate choice that an empty Go string
// maps to a nil C pointer (the native API treats NULL as "unset"), and that
// CGoString round-trips that back to "".
func TestCStringEmptyIsNil(t *testing.T) {
	if c := CString(""); c != nil {
		t.Errorf("CString(\"\") = %p, want nil", c)
	}
	if got := CGoString(nil); got != "" {
		t.Errorf("CGoString(nil) = %q, want empty string", got)
	}
}

// TestCGoStringStopsAtNUL ensures CGoString reads only up to the first NUL and
// not trailing bytes that may follow in the backing buffer.
func TestCGoStringStopsAtNUL(t *testing.T) {
	buf := []uint8{'h', 'i', 0, 'X', 'Y'}
	if got := CGoString(&buf[0]); got != "hi" {
		t.Fatalf("CGoString = %q, want %q", got, "hi")
	}
}
