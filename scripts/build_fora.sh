#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORA_ROOT="$REPO_ROOT/third_party/fora"
BUILD_DIR="$FORA_ROOT/build"
BOOST_ROOT="$REPO_ROOT/third_party/boost_release"
BOOST_STAGE_LIB="$BOOST_ROOT/stage/lib"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found" >&2; exit 1; }
command -v make >/dev/null 2>&1 || { echo "make not found" >&2; exit 1; }

mkdir -p "$BUILD_DIR"
if [[ -d "$BOOST_ROOT" ]]; then
  cmake -S "$FORA_ROOT" -B "$BUILD_DIR" -DBOOST_ROOT="$BOOST_ROOT" -DBoost_NO_SYSTEM_PATHS=ON -DBOOST_INCLUDEDIR="$BOOST_ROOT" -DBOOST_LIBRARYDIR="$BOOST_STAGE_LIB"
else
  cmake -S "$FORA_ROOT" -B "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" -- -j

if [[ -x "$FORA_ROOT/fora" ]]; then
  cp "$FORA_ROOT/fora" "$BUILD_DIR/fora"
fi

if [[ ! -x "$BUILD_DIR/fora" ]]; then
  echo "FORA binary not found at $BUILD_DIR/fora after build" >&2
  exit 1
fi

echo "Built FORA at $BUILD_DIR/fora"
