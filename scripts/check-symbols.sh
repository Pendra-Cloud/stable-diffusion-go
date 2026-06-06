#!/usr/bin/env bash
# check-symbols.sh — fail-closed verification that a compiled
# stable-diffusion shared library exports exactly the symbol set the Go
# binding (pkg/sd/load.go) registers, and that no ggml symbol leaked.
#
# Usage:
#   scripts/check-symbols.sh <path-to-lib> [expected-symbols.txt]
#
# It asserts BOTH:
#   (a) every symbol listed in expected-symbols.txt is present and exported, and
#   (b) no exported symbol begins with `ggml_` — ggml must be statically linked
#       with hidden visibility so it never enters the global dynamic scope and
#       collides with another in-process ggml-bearing backend.
#
# Works on Linux (nm -D) and macOS (nm -gU, which prefixes a leading `_` that
# we strip). Windows uses dumpbin /EXPORTS directly in the workflow.
set -euo pipefail

lib="${1:?usage: check-symbols.sh <lib> [expected-symbols.txt]}"
expected="${2:-$(dirname "$0")/../lib/expected-symbols.txt}"

if [ ! -f "$lib" ]; then
  echo "check-symbols: library not found: $lib" >&2
  exit 1
fi
if [ ! -f "$expected" ]; then
  echo "check-symbols: expected-symbols file not found: $expected" >&2
  exit 1
fi

uname_s="$(uname -s)"
case "$uname_s" in
  # macOS BSD nm: -g external only, -U defined only.
  Darwin) nm_cmd=(nm -gU "$lib") ;;
  # GNU nm: -D dynamic symbols, --defined-only drops undefined/imported entries
  # (type U) so the gate inspects only true exports — an imported ggml_* (or a
  # never-resolved sd_*) can't false-pass or false-leak.
  *)      nm_cmd=(nm -D --defined-only "$lib") ;;
esac

# Exported symbol names, one per line, with any leading underscore (macOS
# mangling) stripped so the names match the C identifiers in the binding.
exported="$("${nm_cmd[@]}" 2>/dev/null | awk '{print $NF}' | sed 's/^_//' | sort -u)"

missing=0
while IFS= read -r sym; do
  [ -z "$sym" ] && continue
  if ! printf '%s\n' "$exported" | grep -qx "$sym"; then
    echo "MISSING export: $sym" >&2
    missing=$((missing + 1))
  fi
done < "$expected"

# ggml_metallib_start / ggml_metallib_end are linker-defined data markers that
# bracket the Metal shader blob embedded into the macOS build. They carry no
# ggml compute API, and macOS's two-level namespace binds each dylib's
# references to its own copy, so they can't cross-bind with another in-process
# ggml. Allow them ONLY on Darwin (they can't legitimately appear on Linux, so
# the gate stays maximally strict there); still fail on any real ggml_* leak.
if [ "$uname_s" = "Darwin" ]; then
  leaked="$(printf '%s\n' "$exported" | grep -E '^ggml_' | grep -Ev '^ggml_metallib_(start|end)$' || true)"
else
  leaked="$(printf '%s\n' "$exported" | grep -E '^ggml_' || true)"
fi
leak_count=0
if [ -n "$leaked" ]; then
  leak_count="$(printf '%s\n' "$leaked" | wc -l | tr -d ' ')"
  echo "LEAKED ggml symbols (ggml must be static + hidden):" >&2
  printf '%s\n' "$leaked" | sed 's/^/  /' >&2
fi

if [ "$missing" -ne 0 ] || [ "$leak_count" -ne 0 ]; then
  echo "check-symbols: FAILED ($missing missing, $leak_count leaked) for $lib" >&2
  exit 1
fi

echo "check-symbols: OK — all $(grep -cve '^[[:space:]]*$' "$expected") expected symbols exported, no ggml leakage ($lib)"
