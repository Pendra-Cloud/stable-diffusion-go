package sd

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"
)

// registerLibFuncRE matches the symbol-name string literal in each
// purego.RegisterLibFunc(&fn, libSD, "symbol_name") call in load.go.
var registerLibFuncRE = regexp.MustCompile(`RegisterLibFunc\([^,]+,\s*libSD,\s*"([a-zA-Z0-9_]+)"\)`)

// symbolsFromLoadGo returns the sorted, de-duplicated set of native symbol
// names registerFunctions binds, parsed straight from load.go's source.
func symbolsFromLoadGo(t *testing.T) []string {
	t.Helper()
	src, err := os.ReadFile("load.go")
	if err != nil {
		t.Fatalf("read load.go: %v", err)
	}
	matches := registerLibFuncRE.FindAllStringSubmatch(string(src), -1)
	if len(matches) == 0 {
		t.Fatal("no RegisterLibFunc symbols found in load.go — parser out of date?")
	}
	seen := make(map[string]struct{}, len(matches))
	var out []string
	for _, m := range matches {
		if _, dup := seen[m[1]]; dup {
			continue
		}
		seen[m[1]] = struct{}{}
		out = append(out, m[1])
	}
	sort.Strings(out)
	return out
}

// expectedSymbols returns the sorted, de-duplicated contents of
// lib/expected-symbols.txt (blank lines and leading/trailing space ignored).
func expectedSymbols(t *testing.T) []string {
	t.Helper()
	path := filepath.Join("..", "..", "lib", "expected-symbols.txt")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	seen := make(map[string]struct{})
	var out []string
	for _, line := range splitLines(string(data)) {
		if line == "" {
			continue
		}
		if _, dup := seen[line]; dup {
			continue
		}
		seen[line] = struct{}{}
		out = append(out, line)
	}
	sort.Strings(out)
	return out
}

// splitLines splits on \n and trims whitespace/carriage returns so the file
// compares identically regardless of platform line endings.
func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == '\n' {
			line := s[start:i]
			// trim trailing CR and surrounding spaces/tabs.
			for len(line) > 0 && (line[len(line)-1] == '\r' || line[len(line)-1] == ' ' || line[len(line)-1] == '\t') {
				line = line[:len(line)-1]
			}
			for len(line) > 0 && (line[0] == ' ' || line[0] == '\t') {
				line = line[1:]
			}
			out = append(out, line)
			start = i + 1
		}
	}
	return out
}

// TestExpectedSymbolsMatchBinding is the drift guard: the committed
// lib/expected-symbols.txt — which the build-libs.yml symbol-verification gate
// asserts the compiled library exports — MUST list exactly the symbols
// registerFunctions binds. If they diverge, either the binding gained/lost a
// symbol without the libs being rebuilt to export it (Load would fail at
// runtime), or the gate would check a stale set. Regenerate the file with:
//
//	grep -oE 'libSD, "[a-z0-9_]+"' pkg/sd/load.go | grep -oE '"[a-z0-9_]+"' | tr -d '"' | sort -u > lib/expected-symbols.txt
func TestExpectedSymbolsMatchBinding(t *testing.T) {
	got := symbolsFromLoadGo(t)
	want := expectedSymbols(t)

	if len(got) != len(want) {
		t.Errorf("symbol count mismatch: load.go has %d, expected-symbols.txt has %d", len(got), len(want))
	}

	gotSet := toSet(got)
	wantSet := toSet(want)
	for _, s := range got {
		if _, ok := wantSet[s]; !ok {
			t.Errorf("symbol %q is registered in load.go but missing from lib/expected-symbols.txt", s)
		}
	}
	for _, s := range want {
		if _, ok := gotSet[s]; !ok {
			t.Errorf("symbol %q is in lib/expected-symbols.txt but no longer registered in load.go", s)
		}
	}
}

func toSet(ss []string) map[string]struct{} {
	m := make(map[string]struct{}, len(ss))
	for _, s := range ss {
		m[s] = struct{}{}
	}
	return m
}
