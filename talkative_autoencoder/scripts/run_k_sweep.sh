#\!/bin/bash
# Wrapper script for K-sweep evaluation
# Passes all arguments to submit_k_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/submit_k_sweep.sh" "$@"
