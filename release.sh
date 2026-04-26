#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 patch|minor|major" >&2; exit 1; }

[[ $# -eq 1 ]] || usage
bump="$1"
[[ "$bump" =~ ^(patch|minor|major)$ ]] || usage

current=$(python3 -c "
import re, sys
m = re.search(r'version = \"([^\"]+)\"', open('pyproject.toml').read())
print(m.group(1)) if m else sys.exit(1)
")

new=$(python3 -c "
parts = '$current'.split('.')
major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
if '$bump' == 'major': major += 1; minor = 0; patch = 0
elif '$bump' == 'minor': minor += 1; patch = 0
else: patch += 1
print(f'{major}.{minor}.{patch}')
")

echo "$current -> $new"

sed -i "s/^version = \"$current\"/version = \"$new\"/" pyproject.toml
uv lock

git add pyproject.toml uv.lock
git commit -m "bump version to $new"
git tag "v$new"

echo "tagged v$new"
