#!/bin/bash
# filepath: /root/xiaolong/run_profile_avg.sh

OUTPUT_FILE="profile_results_avg_10.csv"

# Write CSV Header
echo "TokenLength,HiddenSize,Experts,Fused_CPU_Avg_ms,Fused_XPU_Avg_ms,Grouped_CPU_Avg_ms,Grouped_XPU_Avg_ms" > "$OUTPUT_FILE"

TOKEN_LENGTHS=(8 16 128 256 512 4096 8192 16372 50000 100000)
HIDDEN_SIZE=712
EXPERTS=256
NUM_RUNS=10

echo "Starting profiling run (Average of $NUM_RUNS runs)..."
echo "------------------------------------------------"

# Helper function to parse time using python
parse_time_py() {
    python3 -c "
import sys, re
val = sys.argv[1]
if not val or val == 'N/A':
    print(0.0)
    sys.exit()
m = re.match(r'([0-9.]+)([a-z]+)', val)
if m:
    n = float(m.group(1))
    u = m.group(2)
    # Normalize everything to milliseconds
    factor = 1.0
    if u == 'us': factor = 1
    elif u == 'ms': factor = 1000.0
    print(n * factor)
else:
    print(0.0)
" "$1"
}

for T in "${TOKEN_LENGTHS[@]}"; do
    echo "Processing Token Length: $T"
    
    # Reset accumulators for this token length
    # We store the running sum in milliseconds (float)
    sum_fused_cpu=0.0
    sum_fused_xpu=0.0
    sum_grouped_cpu=0.0
    sum_grouped_xpu=0.0
    
    for ((i=1; i<=NUM_RUNS; i++)); do
        # Capture stdout & stderr combined
        # Use taskset to pin to core 0 to reduce variability
        OUTPUT=$(taskset -c 0 python test_fused_grouped_topk.py -t "$T" -h "$HIDDEN_SIZE" -e "$EXPERTS" --profile 2>&1)
        
        # 1. Parse FUSED KERNEL (Col 6=CPU, Col 10=XPU)
        LINE_F=$(echo "$OUTPUT" | grep "sycl_fused_grouped_topk::fused_grouped_topk" | head -n 1)
        RAW_F_CPU=$(echo "$LINE_F" | awk '{print $6}')
        RAW_F_XPU=$(echo "$LINE_F" | awk '{print $10}')
        
        # 2. Parse GROUPED KERNEL (Col 6=CPU, Col 10=XPU)
        LINE_G=$(echo "$OUTPUT" | grep "sycl_fused_grouped_topk::grouped_topk" | head -n 1)
        RAW_G_CPU=$(echo "$LINE_G" | awk '{print $6}')
        RAW_G_XPU=$(echo "$LINE_G" | awk '{print $10}')

        # Clean raw strings (e.g. handle empty greps)
        if [ -z "$RAW_F_CPU" ]; then RAW_F_CPU="0us"; fi
        if [ -z "$RAW_F_XPU" ]; then RAW_F_XPU="0us"; fi
        if [ -z "$RAW_G_CPU" ]; then RAW_G_CPU="0us"; fi
        if [ -z "$RAW_G_XPU" ]; then RAW_G_XPU="0us"; fi

        # Convert to milliseconds float
        VAL_F_CPU=$(python3 -c "print($(parse_time_py "$RAW_F_CPU"))")
        VAL_F_XPU=$(python3 -c "print($(parse_time_py "$RAW_F_XPU"))")
        VAL_G_CPU=$(python3 -c "print($(parse_time_py "$RAW_G_CPU"))")
        VAL_G_XPU=$(python3 -c "print($(parse_time_py "$RAW_G_XPU"))")
        
        # Accumulate
        sum_fused_cpu=$(python3 -c "print($sum_fused_cpu + $VAL_F_CPU)")
        sum_fused_xpu=$(python3 -c "print($sum_fused_xpu + $VAL_F_XPU)")
        sum_grouped_cpu=$(python3 -c "print($sum_grouped_cpu + $VAL_G_CPU)")
        sum_grouped_xpu=$(python3 -c "print($sum_grouped_xpu + $VAL_G_XPU)")
        
        # Print progress with BOTH CPU and XPU times
        echo -ne "  [Run $i/$NUM_RUNS] Fused(CPU/XPU): ${RAW_F_CPU}/${RAW_F_XPU} | Grouped(CPU/XPU): ${RAW_G_CPU}/${RAW_G_XPU}    \r"
    done
    
    # Calculate Averages
    AVG_F_CPU=$(python3 -c "print(f'{($sum_fused_cpu / $NUM_RUNS):.4f}')")
    AVG_F_XPU=$(python3 -c "print(f'{($sum_fused_xpu / $NUM_RUNS):.4f}')")
    AVG_G_CPU=$(python3 -c "print(f'{($sum_grouped_cpu / $NUM_RUNS):.4f}')")
    AVG_G_XPU=$(python3 -c "print(f'{($sum_grouped_xpu / $NUM_RUNS):.4f}')")

    echo "" # New line to clear carriage return
    echo "  => Avg (ms): Fused_CPU=$AVG_F_CPU Fused_XPU=$AVG_F_XPU | Grouped_CPU=$AVG_G_CPU Grouped_XPU=$AVG_G_XPU"
    
    # Append to CSV
    echo "$T,$HIDDEN_SIZE,$EXPERTS,$AVG_F_CPU,$AVG_F_XPU,$AVG_G_CPU,$AVG_G_XPU" >> "$OUTPUT_FILE"
    echo "------------------------------------------------"
done

echo "Done. Results saved to $OUTPUT_FILE"
