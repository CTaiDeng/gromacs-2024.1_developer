#!/usr/bin/env bash
set -euo pipefail

# 说明：
# - 在 WSL(Ubuntu/Debian) 下，从 SMILES 或输入结构文件生成标准的 hiv.mol2
# - 优先通过 Open Babel 生成3D坐标（SDF），再用 antechamber 输出 Tripos mol2（含 SUBSTRUCTURE、残基名）
# - 需要的工具：antechamber（AmberTools），建议安装 openbabel（处理 SMILES/2D 转 3D）
#
# 先决条件（conda 环境示例）：
#   conda create -n amber -c conda-forge ambertools openbabel -y
#   conda activate amber
#
# 用法：
#   bash my_scripts/generate_hiv_mol2_wsl.sh [选项]
#
# 选项：
#   --input <file>     输入结构文件（mol2/sdf/pdb/mol 等），若是2D建议安装 openbabel
#   --smiles <str>     直接指定 SMILES 字符串（需 openbabel）
#   --resname <name>   残基名（默认：LIG）
#   --charge <int>     净电荷（默认：0）
#   --out <file>       输出文件（默认：hiv.mol2）
#   --ph <float>       生成3D时假定的 pH（默认：7.4，仅 openbabel 用）
#   -h, --help         显示帮助
#
# 示例：
#   # 从 SMILES 直接生成 hiv.mol2（中性，残基名 LIG）
#   bash my_scripts/generate_hiv_mol2_wsl.sh --smiles "c1ccccc1C(=O)N" --charge 0 --resname LIG
#
#   # 从现有文件（如 lig.pdb 或 lig.mol2）生成 hiv.mol2
#   bash my_scripts/generate_hiv_mol2_wsl.sh --input lig.pdb --charge 0 --resname LIG

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERR] 缺少依赖：$1" >&2; exit 1; }
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

usage() {
  sed -n '1,60p' "$0" | sed -n '/^# 用法/,/^need_cmd/p' | sed 's/^# \{0,1\}//'
}

INPUT=""
SMILES=""
RESNAME="LIG"
CHARGE="0"
OUT="hiv.mol2"
PH="7.4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)  INPUT="$2"; shift 2;;
    --smiles) SMILES="$2"; shift 2;;
    --resname) RESNAME="$2"; shift 2;;
    --charge) CHARGE="$2"; shift 2;;
    --out)    OUT="$2"; shift 2;;
    --ph)     PH="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERR] 未知参数：$1" >&2; usage; exit 1;;
  esac
done

# 基础依赖：antechamber（来自 ambertools）
need_cmd antechamber

tmpdir=$(mktemp -d)
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

seed_sdf="$tmpdir/seed.sdf"

generate_seed_from_smiles() {
  if ! has_cmd obabel; then
    echo "[ERR] 需要 openbabel 才能从 SMILES 生成3D，请先安装：conda install -c conda-forge openbabel" >&2
    exit 2
  fi
  echo "[INFO] 由 SMILES 生成3D结构 (pH=$PH)"
  obabel -:"$SMILES" -O "$seed_sdf" --gen3d -p "$PH" -h >/dev/null
}

generate_seed_from_file() {
  local in="$1"
  if has_cmd obabel; then
    echo "[INFO] 使用 openbabel 统一转换为3D SDF (pH=$PH)"
    obabel "$in" -O "$seed_sdf" --gen3d -p "$PH" -h >/dev/null
  else
    echo "[WARN] 未检测到 openbabel，尝试直接用 antechamber 处理输入（需文件已含3D坐标）"
    # 直接传给 antechamber（后续用 -fi 自动判定不可靠，这里简单依据扩展名）
    local ext=${in##*.}
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    case "$ext" in
      mol2) seed_sdf="$in";;
      sdf)  seed_sdf="$in";;
      pdb)  seed_sdf="$in";;
      *) echo "[ERR] 无 openbabel 且不支持的输入类型：$ext（建议安装 openbabel）" >&2; exit 3;;
    esac
  fi
}

if [[ -n "$SMILES" ]]; then
  generate_seed_from_smiles
elif [[ -n "$INPUT" ]]; then
  [[ -f "$INPUT" ]] || { echo "[ERR] 输入文件不存在：$INPUT" >&2; exit 1; }
  generate_seed_from_file "$INPUT"
else
  # 无参数时的默认：尝试当前目录 hiv.smi / hiv.sdf / hiv.pdb / hiv.mol2
  if [[ -f hiv.smi ]]; then SMILES=$(cat hiv.smi | head -n1 | awk '{print $1}'); generate_seed_from_smiles; 
  elif [[ -f hiv.sdf ]]; then INPUT=hiv.sdf; generate_seed_from_file "$INPUT";
  elif [[ -f hiv.pdb ]]; then INPUT=hiv.pdb; generate_seed_from_file "$INPUT";
  elif [[ -f hiv.mol2 ]]; then INPUT=hiv.mol2; generate_seed_from_file "$INPUT";
  else
    echo "[ERR] 未提供 --input/--smiles，且目录下无 hiv.smi/sdf/pdb/mol2" >&2
    usage
    exit 1
  fi
fi

echo "[INFO] 使用 antechamber 生成 Tripos mol2（GAFF2 + BCC电荷）"
antechamber -i "$seed_sdf" -fi sdf \
            -o "$tmpdir/lig.mol2" -fo mol2 \
            -rn "$RESNAME" -at gaff2 -c bcc -nc "$CHARGE" -dr no >/dev/null

# 简单校验：是否存在 SUBSTRUCTURE
if ! grep -q '^@<TRIPOS>SUBSTRUCTURE' "$tmpdir/lig.mol2"; then
  echo "[ERR] 生成的 mol2 缺少 SUBSTRUCTURE 段，可能输入缺少3D或依赖异常。" >&2
  echo "      请确认 openbabel 与 ambertools 安装，并确保已生成3D坐标。" >&2
  exit 4
fi

mv -f "$tmpdir/lig.mol2" "$OUT"
echo "[DONE] 已生成：$OUT"
echo "       残基名：$RESNAME，净电荷：$CHARGE"
echo "       可用于 acpype：acpype -i $OUT -b $RESNAME -o gmx"

