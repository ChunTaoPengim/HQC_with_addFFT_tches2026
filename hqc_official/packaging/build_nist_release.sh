#!/usr/bin/env bash
# packaging/build_nist_release.sh
# Assemble NIST‐style release under <repo-root>/build-package:
#  - Reference_Implementation/{hqc-1,hqc-3,hqc-5}
#  - Optimized_Implementation/{hqc-1,hqc-3,hqc-5} (only AVX256)
#  - KATs (ref → Reference_Implementation, avx256 → Optimized_Implementation)
#  - Supporting_Documentation placeholder
#  - …and all the docs, sources, Makefiles, LICENSE, libs, helpers, etc.

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
UTILS_DIR="$SCRIPT_DIR/utils"
DIST_DIR="$REPO_ROOT/build-package"

# Clean previous release
if [ -d "$DIST_DIR" ]; then
  echo "Removing existing $DIST_DIR/"
  rm -rf "$DIST_DIR"
fi

# Create top‐level
mkdir -p \
  "$DIST_DIR" \
  "$DIST_DIR/Reference_Implementation" \
  "$DIST_DIR/Optimized_Implementation" \
  "$DIST_DIR/KATs" \
  "$DIST_DIR/Supporting_Documentation"

# Copy top‐level README if present
[ -f "$UTILS_DIR/README" ] && cp "$UTILS_DIR/README" "$DIST_DIR/README"

# Copy CHANGELOG if present
[ -f "$UTILS_DIR/CHANGELOG.md" ] && cp "$UTILS_DIR/CHANGELOG.md" "$DIST_DIR/CHANGELOG.md"

VARIANTS=(hqc-1 hqc-3 hqc-5)

generate_impl() {
  local ARCH="$1"         # for docs: "ref" or "x86_64"
  local SRC_ARCH="$2"     # where to pull code: "ref" or "x86_64"
  local OUT_DIR="$3"      # "Reference_Implementation" or "Optimized_Implementation"
  local README_IN="$4"
  local MAKEFILE_IN="$5"
  local MICRO_ARCH="${6-}"  # e.g. "avx256" for optimized

  echo "=== Populating $OUT_DIR ==="
  [ -f "$UTILS_DIR/$README_IN" ] && cp "$UTILS_DIR/$README_IN" "$DIST_DIR/$OUT_DIR/README"

  for VAR in "${VARIANTS[@]}"; do
    NUM=${VAR#hqc-}
    DST="$DIST_DIR/$OUT_DIR/$VAR"
    mkdir -p "$DST/doc" "$DST/src" "$DST/lib"

    # Documentation + Doxygen
    cp "$UTILS_DIR/docs/biblio.bib"           "$DST/doc/"
    cp "$UTILS_DIR/docs/$ARCH/main_page.txt"  "$DST/doc/"
    cp "$UTILS_DIR/doxygen/doxygen.conf"      "$DST/"

    # Makefile → replace xxx with variant number
    cp "$UTILS_DIR/make/$MAKEFILE_IN" "$DST/Makefile"
    chmod +x "$DST/Makefile"
    sed -i "s/xxx/$NUM/g" "$DST/Makefile"

    # 1) common code
    cp "$REPO_ROOT/src/common/"*.c "$REPO_ROOT/src/common/"*.h "$DST/src/"

    # 2) exactly one api.h for this VAR
    cp "$REPO_ROOT/src/common/$VAR/api.h" "$DST/src/"

    if [ "$SRC_ARCH" = "ref" ]; then
      # reference: core + per-variant
      cp "$REPO_ROOT/src/ref/"*.c "$REPO_ROOT/src/ref/"*.h "$DST/src/"
      cp "$REPO_ROOT/src/ref/$VAR/"* "$DST/src/"
    else
      # optimized: x86_64/common + micro-arch + per-variant
      cp "$REPO_ROOT/src/x86_64/common/"*.c "$REPO_ROOT/src/x86_64/common/"*.h "$DST/src/"
      cp "$REPO_ROOT/src/x86_64/common/$VAR/"* "$DST/src/"
      cp "$REPO_ROOT/src/x86_64/$MICRO_ARCH/"*.c "$REPO_ROOT/src/x86_64/$MICRO_ARCH/"*.h "$DST/src/"
      cp "$REPO_ROOT/src/x86_64/$MICRO_ARCH/$VAR/"* "$DST/src/"
    fi

    # helpers, license, libs
    cp "$UTILS_DIR/helpers/main_hqc.c" "$DST/src/"
    cp "$UTILS_DIR/helpers/main_kat.c" "$DST/src/"
    [ -f "$REPO_ROOT/LICENSE" ] && cp "$REPO_ROOT/LICENSE" "$DST/"
    for entry in "$REPO_ROOT/lib/"*; do
      [[ $(basename "$entry") != CMakeLists.txt ]] && cp -r "$entry" "$DST/lib/"
    done
  done
}

# Build Reference (full ref)
generate_impl ref  ref    Reference_Implementation  README_ref    Makefile_ref

# Build Optimized (only AVX256)
generate_impl x86_64 x86_64 Optimized_Implementation README_x86_64 Makefile_x86_64 avx256

# Copy KATs
echo "Copying KATs…"
rm -rf "$DIST_DIR/KATs"
mkdir -p \
  "$DIST_DIR/KATs/Reference_Implementation" \
  "$DIST_DIR/KATs/Optimized_Implementation"

cp -r "$REPO_ROOT/kats/ref/"*            "$DIST_DIR/KATs/Reference_Implementation/"
cp -r "$REPO_ROOT/kats/x86_64/avx256/"*  "$DIST_DIR/KATs/Optimized_Implementation/"

echo "Release assembled in $DIST_DIR/"
