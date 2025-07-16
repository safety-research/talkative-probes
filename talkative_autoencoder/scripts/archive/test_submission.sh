#!/bin/bash
# Test script to verify SLURM submission works

echo "=== Testing SLURM Submission ==="

# Test 1: Minimal test job
echo -e "\n1. Testing minimal job submission..."
JOB_ID=$(sbatch --parsable scripts/slurm_minimal_test.sh 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Success! Job ID: $JOB_ID"
else
    echo "✗ Failed: $JOB_ID"
    echo "Trying with even more minimal options..."
    
    # Try inline submission
    JOB_ID=$(sbatch --parsable --job-name=test --gres=gpu:1 --wrap="echo test; nvidia-smi" 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "✓ Inline submission worked! Job ID: $JOB_ID"
    else
        echo "✗ Inline also failed: $JOB_ID"
    fi
fi

# Test 2: Check if we can submit without time limit
echo -e "\n2. Testing without time limit..."
cat > /tmp/test_notime.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test-notime
#SBATCH --gres=gpu:1
#SBATCH --nodelist=330702be7061
echo "Test without time limit"
nvidia-smi
EOF
chmod +x /tmp/test_notime.sh

JOB_ID=$(sbatch --parsable /tmp/test_notime.sh 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Works without time limit! Job ID: $JOB_ID"
else
    echo "✗ Time limit might be required: $JOB_ID"
fi

# Test 3: Check current jobs
echo -e "\n3. Current jobs:"
squeue -u $USER

echo -e "\n=== Recommendations ==="
echo "Based on the tests above, adjust the SLURM scripts accordingly."
echo "The minimal working configuration will be shown above."