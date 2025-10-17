#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# 本脚本为自由软件，遵循 GPL-3.0；不提供任何担保。
# 目的：在 WSL/Ubuntu 安装 Python3 开发组件（头文件/库），用于修复
#       CMake “Could NOT find Python3 (missing: ... Development[.Module|.Embed])”。

set -euo pipefail

show_help() {
  cat <<'EOF'
在 WSL/Ubuntu 安装 Python3 开发组件以满足 CMake 的 FindPython3(… COMPONENTS Development …)。

用法：
  bash my_scripts/install_python_dev_wsl.sh [--py 3.X] [--dry-run]

选项：
  --py 3.X   指定目标 Python 次版本（如 3.10）。默认自动检测系统 python3。
  --dry-run  仅显示将安装的包，不实际执行。

说明：
  - 基础包：python3、python3-pip、python3-venv、python3-dev、pkg-config。
  - 若存在版本特定包：python3.X-dev、libpython3.X-dev，也会一并安装。
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  show_help; exit 0
fi

PYVER_SPEC=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --py) PYVER_SPEC="${2:?}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) show_help; exit 0;;
    *) echo "[WARN] 未知参数：$1" >&2; shift;;
  esac
done

detected_pyver() {
  if [[ -n "${PYVER_SPEC}" ]]; then
    echo "${PYVER_SPEC}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
    return 0
  fi
  echo "3"
}

apt_has_pkg() {
  local pkg="$1"
  apt-cache policy "$pkg" >/dev/null 2>&1
}

main() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[ERROR] 未找到 apt-get，仅支持 WSL/Ubuntu 环境。" >&2
    exit 2
  fi

  local ver pkgs extras=()
  ver="$(detected_pyver)"
  pkgs=(python3 python3-pip python3-venv python3-dev pkg-config)

  local ver_pkg="python${ver}-dev"
  local libver_pkg="libpython${ver}-dev"
  if apt_has_pkg "$ver_pkg"; then extras+=("$ver_pkg"); fi
  if apt_has_pkg "$libver_pkg"; then extras+=("$libver_pkg"); fi

  echo "[INFO] Python3 目标版本：${ver}"
  echo "[INFO] 基础包：${pkgs[*]}"
  if ((${#extras[@]})); then
    echo "[INFO] 版本特定包：${extras[*]}"
  else
    echo "[INFO] 未发现版本特定包（$ver），将仅安装基础包。"
  fi

  if (( DRY_RUN )); then
    echo "[DRY-RUN] sudo apt-get update -y"
    echo "[DRY-RUN] sudo apt-get install -y ${pkgs[*]} ${extras[*]:-}"
    exit 0
  fi

  sudo apt-get update -y
  sudo apt-get install -y "${pkgs[@]}" ${extras[*]:-}

  echo "[OK] Python3 开发组件已安装。若 CMake 仍失败，可在配置时追加："
  echo "     -DPython3_FIND_STRATEGY=LOCATION -DPython3_FIND_IMPLEMENTATIONS=CPython"
}

main "$@"

