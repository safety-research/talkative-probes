#!/bin/bash
# Script to get an interactive GPU session with SLURM

# Default values
TIME="2:00:00"
#MEM="32G"
GPUS=1
PARTITION="gpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -m|--mem)
            MEM="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -t, --time TIME       Allocation time (default: 2:00:00)"
            echo "  -m, --mem MEMORY      Memory allocation (default: 32G)"
            echo "  -g, --gpus NUM        Number of GPUs (default: 1)"
            echo "  -p, --partition NAME  SLURM partition (default: gpu)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --time 4:00:00 --mem 64G --gpus 1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Requesting interactive GPU session..."
echo "  Partition: $PARTITION"
echo "  GPUs: $GPUS"
echo "  Time: $TIME"
echo "  Memory: $MEM"
echo ""

# Request interactive session
salloc  --gres=gpu:"$GPUS" #--time="$TIME" --mem="$MEM"