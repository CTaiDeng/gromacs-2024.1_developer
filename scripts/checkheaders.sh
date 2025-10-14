#!/usr/bin/env bash
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

incldir="../include"
rm -f gromacs
ln -s $incldir gromacs

if [ -z "$1" ]; then
  files=$(cd $incldir; find -name "*.h" | sed 's/^\./gromacs/')
else
  files="$@"
fi

for i in $files; do
  echo $i
  cat << EOF > t.c
#include <$i>
int main(){
  return 0;
}
EOF
  gcc -I. -c t.c -D bool=int || echo "Failed"
done
rm -f gromacs t.[co]
