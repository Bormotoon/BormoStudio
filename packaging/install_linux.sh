#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! -f $1 ]]; then
  printf 'Usage: bash packaging/install_linux.sh /path/to/bormostudio\n' >&2
  exit 2
fi
binary=$1
prefix=${BSTUDIO_INSTALL_PREFIX:-${UVF_INSTALL_PREFIX:-"$HOME/.local"}}
mkdir -p "$prefix/bin" "$prefix/share/applications"
install -m 755 "$binary" "$prefix/bin/bormostudio"
ln -sfn bormostudio "$prefix/bin/bstudio"
# The pre-rename command keeps working.
ln -sfn bormostudio "$prefix/bin/uvf"
desktop="$prefix/share/applications/bormostudio.desktop"
printf '[Desktop Entry]\nType=Application\nName=BormoStudio\nExec="%s/bin/bormostudio" --gui\nTerminal=false\nCategories=AudioVideo;Video;\n' "$prefix" > "$desktop"
"$prefix/bin/bstudio" --check-resources
printf 'Installed to %s/bin; add this directory to PATH if needed.\n' "$prefix"
