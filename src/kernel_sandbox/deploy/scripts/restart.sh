#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config="${1:?usage: restart.sh /absolute/path/to/master.yaml}"
"$root/deploy/scripts/master.sh" restart "$config"
