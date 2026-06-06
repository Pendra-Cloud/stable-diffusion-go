package sd

import "testing"

// TestLoadMissingLibReturnsError verifies that loading from a directory with no
// shared library returns an error instead of panicking, so importers stay
// healthy when the native lib is absent.
func TestLoadMissingLibReturnsError(t *testing.T) {
	// The premise only holds while the library hasn't already been loaded in
	// this process; if it has, Load is a no-op and returns nil by design.
	if loaded {
		t.Skip("library already loaded in this process; missing-lib path not exercisable")
	}
	if err := Load(t.TempDir()); err == nil {
		t.Fatal("expected an error loading from an empty dir, got nil")
	}
	if loaded {
		t.Fatal("loaded flag should remain false after a failed Load")
	}
}

// TestLibCandidatesEmptyDir verifies an empty libDir resolves to the bare
// filename so the OS default search path is used.
func TestLibCandidatesEmptyDir(t *testing.T) {
	got := libCandidates("")
	if len(got) != 1 || got[0] != libFileName() {
		t.Fatalf("expected [%q], got %v", libFileName(), got)
	}
}
