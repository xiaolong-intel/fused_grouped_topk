# docker image

# proxy setting
```
export http_proxy=http://child-prc.intel.com:913
export https_proxy=http://child-prc.intel.com:913
export no_proxy=localhost,127.0.0.1,10.0.0.0/8,.intel.com
```
# Download vllm-xpu-kernels
git clone https://github.com/vllm-project/vllm-xpu-kernels.git
cp csrc/grouped_topk_kernels.cpp vllm-xpu-kernels/csrc/moe/
# Install
pip install -vv -e . --no-build-isolation
# run test
python test_fused_grouped_topk -t token_lenth --profile
# run benchmark
bash bench.sh

```
The main optimized implementation currently lives in <span style="color:red;">vllm-xpu-kernels/csrc/moe/grouped_topk_kernels.cpp(line 660~794)</span>, inside grouped_topk_warp_only_256_kernel(...). \\
This kernel is the dedicated warp-only fast path for the 256-expert grouped top-k case. It is the primary implementation used for performance tuning and correctness debugging in this repository. 
```