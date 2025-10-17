#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# 本脚本为自由软件，遵循 GPL-3.0；不提供任何担保。
# 目的：在 WSL/Ubuntu 安装 Sphinx(>=4.0) 与 Pygments 以满足 CMake 构建文档。

set -euo pipefail

show_help() {
  cat <<'EOF'
安装 Sphinx 文档工具（>=4.0）与 Pygments（语法高亮）。

用法：
  bash my_scripts/install_sphinx_wsl.sh [--venv <PATH> | --system | --apt] [--dry-run]

安装模式（三选一，默认 --venv ./.venv-docs）：
  --venv <PATH>  在指定虚拟环境安装（推荐）。若不存在会自动创建。
  --system       使用 pip 安装到用户环境（--user）。
  --apt          使用 apt 安装系统包（可能版本偏旧，不保证 >=4.0）。

其他参数：
  --dry-run      仅展示将执行的命令。

安装完成后：
  - 若为 venv 模式，sphinx-build 位于 <PATH>/bin/sphinx-build。
  - CMake 可通过 -DSPHINX_EXECUTABLE=<sphinx-build> 指定。
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  show_help; exit 0
fi

MODE="venv"
VENV_PATH=".venv-docs"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) MODE="venv"; VENV_PATH="${2:?}"; shift 2;;
    --system) MODE="system"; shift;;
    --apt) MODE="apt"; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) show_help; exit 0;;
    *) echo "[WARN] 未知参数：$1" >&2; shift;;
  esac
done

run() {
  if (( DRY_RUN )); then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

ensure_python_basics() {
  if ! command -v python3 >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      run "sudo apt-get update -y"
      run "sudo apt-get install -y python3 python3-pip python3-venv"
    else
      echo "[ERROR] 未找到 apt-get，请手动安装 python3/pip3。" >&2
      exit 2
    fi
  fi
}

install_venv() {
  ensure_python_basics
  if [[ ! -d "${VENV_PATH}" ]]; then
    run "python3 -m venv '${VENV_PATH}'"
  fi
  # shellcheck disable=SC1090
  run "source '${VENV_PATH}/bin/activate' && pip install -U pip wheel && pip install -U 'sphinx>=4.0' pygments"
  local exe="${VENV_PATH}/bin/sphinx-build"
  echo "[OK] venv 安装完成：${exe}"
  echo "[HINT] CMake 可使用：-DSPHINX_EXECUTABLE='${exe}'"
}

install_system_user() {
  ensure_python_basics
  run "pip3 install --user -U 'sphinx>=4.0' pygments"
  local exe="$HOME/.local/bin/sphinx-build"
  echo "[OK] 已安装到用户环境：${exe}"
  echo "[HINT] 若命令未找到，请将 ~/.local/bin 加入 PATH。"
}

install_apt() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[ERROR] 未找到 apt-get，仅支持 WSL/Ubuntu 环境。" >&2
    exit 2
  fi
  run "sudo apt-get update -y"
  run "sudo apt-get install -y python3-sphinx python3-pygments"
  local exe="$(command -v sphinx-build || true)"
  echo "[OK] 系统包安装完成：${exe:-未检测到 sphinx-build}"
  echo "[NOTE] 发行版版本可能较旧，如需 >=4.0 建议使用 --venv 或 --system。"
}

main() {
  case "$MODE" in
    venv) install_venv;;
    system) install_system_user;;
    apt) install_apt;;
    *) echo "[ERROR] 未知安装模式：$MODE" >&2; exit 2;;
  esac
}

main "$@"

