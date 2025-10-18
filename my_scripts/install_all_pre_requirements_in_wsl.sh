#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# 本脚本为自由软件，遵循 GPL-3.0；不提供任何担保。
# 目的：统一整合 WSL/Ubuntu 下本仓库常用“预安装/依赖安装/环境修复”脚本，
#       一键或按需安装 Python 开发组件、Sphinx 文档依赖、CMake 等。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

run() {
  if (( DRY_RUN )); then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

show_help() {
  cat <<'EOF'
统一安装脚本（WSL/Ubuntu）
—— 集成以下子脚本：
  - install_python_dev_wsl.sh    安装 Python3 开发组件
  - install_sphinx_wsl.sh        安装 Sphinx>=4 与 Pygments
  - install_cmake_wsl.sh         安装/升级 CMake（Kitware 源）
  - install_openbabel_wsl.sh     安装 Open Babel（可选）
  - install_amber_deps_wsl.sh    安装 AmberTools 依赖（conda，可选）
  - install_gromacs_wsl.sh       安装 GROMACS（可选）
  - fix_cmake_blas_nvpl_wsl.sh   修复/配置 BLAS/LAPACK 查找（可选）
  - fix_imagemagick_convert_wsl.sh 修复 ImageMagick convert 策略（可选）
  - fix_cmake_python3_dev_wsl.sh 安装 Python 开发并可选择重新 CMake（可选）

用法：
  bash my_scripts/install_all_pre_requirements_in_wsl.sh [选项]

常用场景（无参数默认执行：Python 开发组件 + Sphinx(venv) + CMake）：
  - 默认执行：
      bash my_scripts/install_all_pre_requirements_in_wsl.sh

  - 自定义选择：
      bash my_scripts/install_all_pre_requirements_in_wsl.sh \
        --python-dev --sphinx --cmake \
        [--sphinx-mode venv|system|apt] [--sphinx-venv ./.venv-docs] [--py 3.10]

  - 扩展安装：
      bash my_scripts/install_all_pre_requirements_in_wsl.sh \
        --openbabel --amber-deps --gromacs

  - 构建辅助修复：
      bash my_scripts/install_all_pre_requirements_in_wsl.sh \
        --fix-blas openblas --fix-imagemagick \
        --rerun-cmake --source-dir . --build-dir cmake-build-release-wsl

选项（按需组合）：
  基础/默认：
    --python-dev           安装 Python 开发组件（python3-dev 等）
    --sphinx               安装 Sphinx 与 Pygments（默认 venv 到 ./.venv-docs）
    --cmake                安装/升级 CMake

  额外组件：
    --openbabel            安装 Open Babel
    --amber-deps           安装 AmberTools 依赖（conda）
    --gromacs              安装 GROMACS（可能耗时较长）

  构建修复：
    --fix-blas <mode>      调整 BLAS/NVPL 查找；mode: openblas|internal|suppress
    --fix-imagemagick      修复 ImageMagick convert 安全策略
    --fix-imagemagick-restore  恢复修复前策略
    --rerun-cmake          安装后重新运行 CMake（通过 fix_cmake_python3_dev_wsl.sh）

  细化参数：
    --py 3.X               传给 Python 安装脚本的目标版本（如 3.10）
    --sphinx-mode <m>      venv|system|apt（默认 venv）
    --sphinx-venv <path>   venv 路径（默认 ./.venv-docs）
    --source-dir <DIR>     源码目录（配合 --rerun-cmake）
    --build-dir <DIR>      构建目录（配合 --rerun-cmake）
    --cmake-args "..."     追加传给 CMake 的参数

  通用：
    --all                  等价于 --python-dev --sphinx --cmake
    --dry-run              仅打印将执行的命令
    -h, --help             显示帮助

说明：若未显式选择任何组件，默认等价于 --all。
EOF
}

# 选项默认值
DO_PYDEV=0
DO_SPHINX=0
DO_CMAKE=0
DO_OPENBABEL=0
DO_AMBER=0
DO_GROMACS=0
DO_FIX_BLAS=""
DO_FIX_IMAGE=0
DO_FIX_IMAGE_RESTORE=0
DO_RERUN_CMAKE=0
PYVER_SPEC=""
SPHINX_MODE="venv"
SPHINX_VENV=".venv-docs"
SRC_DIR="${REPO_DIR}"
BUILD_DIR="${REPO_DIR}/cmake-build-release-wsl"
EXTRA_CMAKE_ARGS=""
DRY_RUN=0

if [[ $# -eq 0 ]]; then
  DO_PYDEV=1; DO_SPHINX=1; DO_CMAKE=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DO_PYDEV=1; DO_SPHINX=1; DO_CMAKE=1; shift;;
    --python-dev) DO_PYDEV=1; shift;;
    --sphinx) DO_SPHINX=1; shift;;
    --cmake) DO_CMAKE=1; shift;;
    --openbabel) DO_OPENBABEL=1; shift;;
    --amber-deps) DO_AMBER=1; shift;;
    --gromacs) DO_GROMACS=1; shift;;
    --fix-blas) DO_FIX_BLAS="${2:?需要 mode: openblas|internal|suppress}"; shift 2;;
    --fix-imagemagick) DO_FIX_IMAGE=1; shift;;
    --fix-imagemagick-restore) DO_FIX_IMAGE_RESTORE=1; shift;;
    --rerun-cmake) DO_RERUN_CMAKE=1; shift;;
    --py) PYVER_SPEC="${2:?}"; shift 2;;
    --sphinx-mode) SPHINX_MODE="${2:?}"; shift 2;;
    --sphinx-venv) SPHINX_VENV="${2:?}"; shift 2;;
    --source-dir) SRC_DIR="${2:?}"; shift 2;;
    --build-dir) BUILD_DIR="${2:?}"; shift 2;;
    --cmake-args) EXTRA_CMAKE_ARGS="${2:?}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) show_help; exit 0;;
    *) echo "[WARN] 未知参数：$1" >&2; shift;;
  esac
done

PYDEV_SCRIPT="${REPO_DIR}/my_scripts/install_python_dev_wsl.sh"
SPHINX_SCRIPT="${REPO_DIR}/my_scripts/install_sphinx_wsl.sh"
CMAKE_SCRIPT="${REPO_DIR}/my_scripts/install_cmake_wsl.sh"
OPENBABEL_SCRIPT="${REPO_DIR}/my_scripts/install_openbabel_wsl.sh"
AMBER_SCRIPT="${REPO_DIR}/my_scripts/install_amber_deps_wsl.sh"
GROMACS_SCRIPT="${REPO_DIR}/my_scripts/install_gromacs_wsl.sh"
FIX_BLAS_SCRIPT="${REPO_DIR}/my_scripts/fix_cmake_blas_nvpl_wsl.sh"
FIX_IM_SCRIPT="${REPO_DIR}/my_scripts/fix_imagemagick_convert_wsl.sh"
FIX_CMAKE_PY_SCRIPT="${REPO_DIR}/my_scripts/fix_cmake_python3_dev_wsl.sh"

ensure_exists() {
  local f="$1"
  if [[ ! -x "$f" ]]; then
    echo "[ERROR] 未找到或不可执行：$f" >&2
    exit 2
  fi
}

main() {
  if (( DO_PYDEV )); then
    ensure_exists "$PYDEV_SCRIPT"
    local args=()
    [[ -n "$PYVER_SPEC" ]] && args+=("--py" "$PYVER_SPEC")
    (( DRY_RUN )) && args+=("--dry-run")
    run "bash '$PYDEV_SCRIPT' ${args[*]}"
  fi

  if (( DO_SPHINX )); then
    ensure_exists "$SPHINX_SCRIPT"
    local args=()
    case "$SPHINX_MODE" in
      venv) args+=("--venv" "$SPHINX_VENV");;
      system) args+=("--system");;
      apt) args+=("--apt");;
      *) echo "[ERROR] 非法 --sphinx-mode：$SPHINX_MODE" >&2; exit 2;;
    esac
    (( DRY_RUN )) && args+=("--dry-run")
    run "bash '$SPHINX_SCRIPT' ${args[*]}"
  fi

  if (( DO_CMAKE )); then
    ensure_exists "$CMAKE_SCRIPT"
    (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$CMAKE_SCRIPT\"'" || run "bash '$CMAKE_SCRIPT'"
  fi

  if (( DO_OPENBABEL )); then
    ensure_exists "$OPENBABEL_SCRIPT"
    (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$OPENBABEL_SCRIPT\"'" || run "bash '$OPENBABEL_SCRIPT'"
  fi

  if (( DO_AMBER )); then
    ensure_exists "$AMBER_SCRIPT"
    (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$AMBER_SCRIPT\"'" || run "bash '$AMBER_SCRIPT'"
  fi

  if (( DO_GROMACS )); then
    ensure_exists "$GROMACS_SCRIPT"
    (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$GROMACS_SCRIPT\"'" || run "bash '$GROMACS_SCRIPT'"
  fi

  if [[ -n "$DO_FIX_BLAS" ]]; then
    ensure_exists "$FIX_BLAS_SCRIPT"
    (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$FIX_BLAS_SCRIPT\" --mode $DO_FIX_BLAS'" \
      || run "bash '$FIX_BLAS_SCRIPT' --mode '$DO_FIX_BLAS'"
  fi

  if (( DO_FIX_IMAGE || DO_FIX_IMAGE_RESTORE )); then
    ensure_exists "$FIX_IM_SCRIPT"
    if (( DO_FIX_IMAGE_RESTORE )); then
      (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$FIX_IM_SCRIPT\" --restore'" \
        || run "bash '$FIX_IM_SCRIPT' --restore"
    else
      (( DRY_RUN )) && run "echo '[DRY-RUN] bash \"$FIX_IM_SCRIPT\"'" \
        || run "bash '$FIX_IM_SCRIPT'"
    fi
  fi

  if (( DO_RERUN_CMAKE )); then
    ensure_exists "$FIX_CMAKE_PY_SCRIPT"
    local args=("--source-dir" "$SRC_DIR" "--build-dir" "$BUILD_DIR")
    [[ -n "$EXTRA_CMAKE_ARGS" ]] && args+=("--cmake-args" "$EXTRA_CMAKE_ARGS")
    [[ -n "$PYVER_SPEC" ]] && args+=("--py" "$PYVER_SPEC")
    (( DRY_RUN )) && args+=("--no-apt") && run "echo '[DRY-RUN] bash \"$FIX_CMAKE_PY_SCRIPT\" ${args[*]}'" \
      || run "bash '$FIX_CMAKE_PY_SCRIPT' ${args[*]}"
  fi

  echo "[OK] 预安装/依赖安装步骤完成。"
}

main "$@"
