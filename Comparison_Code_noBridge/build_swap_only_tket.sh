#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE_PYTHON="/Users/tjvo/Desktop/Research/Rice University/Qubit Routing/TketLightSabreCompare/.venv/bin/python"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  "$BASE_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install 'cmake>=3.26' ninja 'conan>=2'
.venv/bin/python -m pip install -r requirements.txt
export PATH="$ROOT/.venv/bin:$PATH"
export CONAN_HOME="$ROOT/.conan2"
conan profile detect --force
if ! conan remote list | grep -q '^tket-libs:'; then
  conan remote add tket-libs https://quantinuumsw.jfrog.io/artifactory/api/conan/tket1-libs --index 0
fi
cd "$ROOT/tket-swap-only"
conan create tket --user=tket --channel=stable --build=missing \
  -o 'boost/*:header_only=True' \
  -o 'tklog/*:shared=True' \
  -o 'tket/*:shared=True' \
  -c tools.build:jobs=2 \
  -tf ''
cd "$ROOT/tket-swap-only/pytket"
SETUPTOOLS_SCM_PRETEND_VERSION=2.18.4 "$ROOT/.venv/bin/python" -m pip install . -v
"$ROOT/.venv/bin/python" -c 'import pytket; print(pytket.__version__, pytket.LEXIROUTE_BRIDGE)'
