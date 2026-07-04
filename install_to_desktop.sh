#!/usr/bin/env bash
set -e
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PARENT="$HOME/桌面/视触觉传感器标定/Gelsight标定"
TARGET_DIR="$TARGET_PARENT/gelsight_force_calibration_project"
mkdir -p "$TARGET_PARENT"
rm -rf "$TARGET_DIR"
cp -a "$SRC_DIR" "$TARGET_DIR"
echo "已复制工程到: $TARGET_DIR"
echo "下一步: cd \"$TARGET_DIR\" && bash scripts/setup_conda_env.sh"
