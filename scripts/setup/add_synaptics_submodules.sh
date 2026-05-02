#!/usr/bin/env bash
# Adds all Synaptics reference repos as submodules under docs/references/upstream/Synaptics/,
# pinned to the exact SHAs recorded in SynapticSL2619 (or nearest remote equivalent for
# the 4 repos whose pinned SHAs are local-only commits that were never pushed).
# Safe to re-run: skips already-registered paths and no-op if SHA already matches.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Target SHA from SynapticSL2619 submodule status
declare -A SHAS=(
  [astra-doc]="d984a968744497606a4e99e65065d767876868f5"       # local-only → use remote HEAD
  [astra-update]="9fb4acd97d98898008570828212c4154c75fecbd"    # local-only → use remote HEAD
  [boot]="5d00892a1e07d80466506f17d1b92ae637d3bb4a"
  [configs]="200ac9250637e5489b63157695e9b1f22daf6135"
  [gstreamer-plugins-syna]="e13cea7337b41a7526a1b385b2c01dafdbd9caf7"
  [linux-drivers-synaptics]="0e07f8a0042be4e636e5280d8e0a3b3a9c3dabfb"
  [sdk]="d5528189ee9ac23037299cbed9f9ca0b2565c7b9"
  [synap-doc]="19adcc29f73cb4cecbc155750d71dfaf02d5bbbb"        # local-only → use remote HEAD
  [synap-examples]="a075494d51a96c5cd16eb2cadd792c25c8015373"
  [synap-framework]="26a1a3a8e78570cef5fb08d84c4b6dc106dd4078"
  [synap-python]="d1e2be54f7745924cdda5538b59176e1691c065d"
  [synap-release]="b1945057032bbc5f530e66d3a2778ba0bc1eae48"
  [synap-rt]="460f3786e5cf01e3fe9a7561947d5507ed9bea84"
  [synap-runtime]="5f7961cfb27a1e09ae77922753793840155264bb"
  [synap-toolkit]="49dc99f509b593e390013bbb85fb2d299e9c9502"
  [torq-compiler]="e69072bbf6df47c7f39a8e63c23772e8945b1420"    # local-only → use remote HEAD
  [torq-examples]="7b460eb027bf09da1c575a8648774a97b557b43d"
  [torq-tools]="f44d39b67192dad691d16510df0d2c4d576ff1dc"
  [usb-tool]="314ca0f0f3ee0c99c56b5f999227a6d72366f1ab"
)

declare -A URLS=(
  [astra-doc]="https://github.com/synaptics-astra/doc.git"
  [astra-update]="https://github.com/synaptics-astra/astra-update"
  [boot]="https://github.com/synaptics-astra/boot.git"
  [configs]="https://github.com/synaptics-astra/configs"
  [gstreamer-plugins-syna]="https://github.com/synaptics-astra/application-gstreamer-plugins-syna"
  [linux-drivers-synaptics]="https://github.com/synaptics-astra/linux_6_12-drivers-synaptics"
  [sdk]="https://github.com/synaptics-astra/sdk.git"
  [synap-doc]="https://github.com/synaptics-synap/doc.git"
  [synap-examples]="https://github.com/synaptics-synap/examples"
  [synap-framework]="https://github.com/synaptics-astra/synap-framework.git"
  [synap-python]="https://github.com/synaptics-synap/synap-python"
  [synap-release]="https://github.com/synaptics-astra/synap-release"
  [synap-rt]="https://github.com/synaptics-synap/synap-rt"
  [synap-runtime]="https://github.com/synaptics-synap/runtime"
  [synap-toolkit]="https://github.com/synaptics-synap/toolkit"
  [torq-compiler]="https://github.com/synaptics-torq/torq-compiler.git"
  [torq-examples]="https://github.com/synaptics-torq/torq-examples.git"
  [torq-tools]="https://github.com/synaptics-torq/torq-tools.git"
  [usb-tool]="https://github.com/synaptics-astra/usb-tool.git"
)

# Branch to fetch when the SHA lives on a non-default branch.
# Omit a key to use the default (main) branch from the initial --depth 1 clone.
declare -A FETCH_BRANCH=(
  [boot]="scarthgap_6.12_v2.3.0"
  [sdk]="scarthgap_6.12_v2.3.0"
  [synap-release]="scarthgap_6.12_v2.3.0"
  [usb-tool]="sl261x"
)

# SHAs that exist only in the SynapticSL2619 local workspace (never pushed upstream).
# We pin these to remote HEAD instead; the difference is recorded in the commit message.
LOCAL_ONLY="astra-doc astra-update synap-doc torq-compiler"

is_local_only() { echo "$LOCAL_ONLY" | grep -qw "$1"; }

TARGET="docs/references/upstream/Synaptics"

pin_to_sha() {
  local path="$1" sha="$2" branch="${3:-}"
  local current
  current="$(git -C "$path" rev-parse HEAD)"
  [[ "$current" == "$sha" ]] && return 0

  echo "    pinning $current -> $sha"

  if [[ -n "$branch" ]]; then
    git -C "$path" fetch --depth=1 origin "$branch"
  fi

  # Try direct SHA fetch first (GitHub supports it when the commit is reachable)
  if git -C "$path" fetch --depth=1 origin "$sha" 2>/dev/null; then
    git -C "$path" checkout --detach "$sha"
    return 0
  fi

  # Already fetched the branch above; check if sha is now reachable
  if [[ -n "$branch" ]] && git -C "$path" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    git -C "$path" checkout --detach "$sha"
    return 0
  fi

  # Fall back: deepen until reachable
  for depth in 50 200 1000; do
    git -C "$path" fetch --depth="$depth" origin 2>/dev/null || true
    if git -C "$path" cat-file -e "${sha}^{commit}" 2>/dev/null; then
      git -C "$path" checkout --detach "$sha"
      return 0
    fi
  done

  echo "    WARNING: SHA $sha not reachable from remote — staying at $(git -C $path rev-parse HEAD)"
}

for name in "${!URLS[@]}"; do
  url="${URLS[$name]}"
  sha="${SHAS[$name]}"
  path="$TARGET/$name"
  branch="${FETCH_BRANCH[$name]:-}"

  echo "==> $name"

  if git config --file .gitmodules "submodule.${path}.url" &>/dev/null; then
    echo "    already registered"
  else
    git submodule add --depth 1 "$url" "$path"
  fi

  if is_local_only "$name"; then
    echo "    NOTE: target SHA is local-only in SynapticSL2619 (never pushed); using remote HEAD"
  else
    pin_to_sha "$path" "$sha" "$branch"
    git add "$path"
  fi
done

# Patch .gitmodules: add update=none, shallow=true, ignore=dirty for every new Synaptics entry
python3 - <<'PY'
import re

with open('.gitmodules') as f:
    content = f.read()

def patch_section(m):
    s = m.group(0)
    if 'docs/references/upstream/Synaptics/' not in s:
        return s
    if 'update =' not in s:
        s = s.rstrip('\n') + '\n\tupdate = none\n'
    if 'shallow =' not in s:
        s = s.rstrip('\n') + '\n\tshallow = true\n'
    if 'ignore =' not in s:
        s = s.rstrip('\n') + '\n\tignore = dirty\n'
    return s

patched = re.sub(
    r'\[submodule "[^"]+"\][^\[]+',
    patch_section,
    content,
    flags=re.DOTALL,
)

with open('.gitmodules', 'w') as f:
    f.write(patched)

print("  .gitmodules patched")
PY

git add .gitmodules
echo ""
echo "Done. Local-only SHA delta (can't be resolved without pushing from SynapticSL2619):"
echo "  astra-doc   : d984a968 (SL2619) vs remote HEAD"
echo "  astra-update: 9fb4acd9 (SL2619) vs remote HEAD"
echo "  synap-doc   : 19adcc29 (SL2619) vs remote HEAD"
echo "  torq-compiler: e69072bb (SL2619) vs remote HEAD"
echo ""
echo "Run 'git submodule status docs/references/upstream/Synaptics/' to verify."
