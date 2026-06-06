//go:build windows

package sd

// bindCFree is intentionally a no-op on Windows.
//
// Image buffers returned by generate_image are allocated by the sd.cpp DLL's C
// runtime. On Windows, freeing memory across CRT heaps (i.e. from a different
// CRT than the one that allocated it) corrupts the heap or crashes, and we
// cannot reliably know which CRT the prebuilt DLL links. Rather than risk that,
// cFree is left nil so FreeImage / FreeImages become safe no-ops here. The
// native sd backend is targeted at Linux/macOS (see BL-102); if Windows support
// lands, bind free from the DLL's own CRT (e.g. the matching ucrtbase/msvcrt).
func bindCFree() {}
