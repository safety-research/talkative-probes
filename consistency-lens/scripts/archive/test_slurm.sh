#!/bin/bash
# Test script to check SLURM configuration

echo "=== SLURM Configuration Test ==="

# Check if SLURM is available
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: sbatch command not found. Are you on a SLURM cluster?"
    exit 1
fi

echo "✓ SLURM commands available"

# Check available partitions
echo -e "\n=== Available Partitions ==="
sinfo -o "%P %a %l %D %N" 2>/dev/null || echo "ERROR: Cannot list partitions"

# Check available constraints
echo -e "\n=== Available Features/Constraints ==="
sinfo -o "%N %f" 2>/dev/null | grep -v "^NODELIST" | sort -u || echo "ERROR: Cannot list features"

# Check current user's default account
echo -e "\n=== Your Account Info ==="
sacctmgr show user $USER 2>/dev/null | grep $USER || echo "Cannot get account info"

# Suggest partition
echo -e "\n=== Suggested SBATCH Options ==="
echo "Based on available partitions, you might need to add:"
DEFAULT_PARTITION=$(sinfo -h -o "%P" | grep '*' | sed 's/*//' | head -1)
if [ -n "$DEFAULT_PARTITION" ]; then
    echo "  Default partition: $DEFAULT_PARTITION"
else
    FIRST_PARTITION=$(sinfo -h -o "%P" | head -1)
    echo "  #SBATCH --partition=$FIRST_PARTITION"
fi

echo -e "\nTo fix the partition error, either:"
echo "1. Add '#SBATCH --partition=<your_partition>' to the scripts"
echo "2. Set SBATCH_PARTITION environment variable: export SBATCH_PARTITION=<your_partition>"
echo "3. Use your cluster's default partition (usually works without specifying)"

echo -e "\n=== Testing Directory ==="
echo "Current directory: $(pwd)"
echo "Project root should be: $(dirname $(dirname $(realpath $0)))"