/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <cmath>
#include <cstdio>
#include <limits>
// #include "../utils.h"
// #include "../dispatch_utils.h"

namespace vllm {
namespace moe {

// Type trait: bfloat16 -> float for computation, everything else stays as-is
template <typename T>
struct compute_type { using type = T; };

template <>
struct compute_type<sycl::ext::oneapi::bfloat16> { using type = float; };

template <typename T>
using compute_type_t = typename compute_type<T>::type;

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
static constexpr int WARP_SIZE = 32;
static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount =
    std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int MaxNumExpertsUnit = 128;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 22;
static constexpr int MaxNumTopGroups = 4;

enum ScoringFunc : int { SCORING_NONE = 0, SCORING_SIGMOID = 1 };

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
class VllmGroupedTopKFusedKernel;

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, bool UseGroups,
          int MaxNumTopExperts = DefaultMaxNumTopExperts>
class VllmGroupedTopKFusedSmallExpertCountKernel;

template <typename T_OUT, typename T_IN>
inline T_OUT sycl_cast(T_IN val) {
    return static_cast<T_OUT>(val);
}

template <>
inline float sycl_cast<float, sycl::half>(sycl::half val) {
    return static_cast<float>(val);
}

template <>
inline float sycl_cast<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
    return static_cast<float>(val);
}

template <typename T>
inline T neg_inf() {
    return sycl_cast<T, float>(-std::numeric_limits<float>::infinity());
}

template <typename T>
inline bool is_finite(const T val) {
    return std::isfinite(sycl_cast<float, T>(val));
}

inline float sigmoid_accurate(float x) {
    return 1.f / (1.f + sycl::native::exp(-x)); // More efficient approximation Optimized point 1
}

template <typename T>
inline T apply_sigmoid(T val) {
    float f = sycl_cast<float, T>(val);
    return sycl_cast<T, float>(sigmoid_accurate(f));
}

template <ScoringFunc SF, typename T>
inline T apply_scoring(T val) {
    if constexpr (SF == SCORING_NONE) {
        return val;
    } else if constexpr (SF == SCORING_SIGMOID) {
        return apply_sigmoid(val);
    } else {
        static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID,
                                    "Unsupported ScoringFunc in apply_scoring");
        return val;
    }
}

namespace warp_topk {

template <int size, typename T>
constexpr T round_up_to_multiple_of(T len) {
    if (len == 0) {
        return 0;
    }
    return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr bool isPowerOf2(T v) {
    return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
inline bool is_better_than(T val, T baseline) {
    return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
inline bool is_better_than(T val, T baseline, idxT index, idxT baseline_index) {
    bool res = (val > baseline && greater) || (val < baseline && !greater);
    if (val == baseline) {
        res = (index < baseline_index && greater) ||
              (index < baseline_index && !greater);
    }
    return res;
}

template <int size, bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge {
    static void merge(T* val_arr, idxT* idx_arr, sycl::nd_item<1> item) {
        static_assert(isPowerOf2(size));
        static_assert(size >= 2 * WARP_SIZE);
        constexpr int arr_len = size / WARP_SIZE;
        constexpr int stride = arr_len / 2;
        for (int i = 0; i < stride; ++i) {
            int const other_i = i + stride;
            T& val = val_arr[i];
            T& other_val = val_arr[other_i];
            bool is_better;
            if constexpr (is_stable) {
                is_better = is_better_than<ascending>(val, other_val, idx_arr[i], idx_arr[other_i]);
            } else {
                is_better = is_better_than<ascending>(val, other_val);
            }
            if (is_better) {
                T tmp = val;
                val = other_val;
                other_val = tmp;
                idxT tmp2 = idx_arr[i];
                idx_arr[i] = idx_arr[other_i];
                idx_arr[other_i] = tmp2;
            }
        }
        BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(val_arr, idx_arr, item);
        BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(val_arr + arr_len / 2, idx_arr + arr_len / 2, item);
    }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
    static void sort(T* val_arr, idxT* idx_arr, sycl::nd_item<1> item) {
        static_assert(isPowerOf2(size));
        static_assert(size >= 2 * WARP_SIZE);
        constexpr int arr_len = size / WARP_SIZE;
        BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr, item);
        BitonicSort<size / 2, false, T, idxT, is_stable>::sort(val_arr + arr_len / 2, idx_arr + arr_len / 2, item);
        BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(val_arr, idx_arr, item);
    }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
    static void sort(T* val_arr, idxT* idx_arr, sycl::nd_item<1> item) {
        int lane = item.get_local_id(0) % WARP_SIZE;
        auto sg = item.get_sub_group();
        for (int stage = 0; stage < 4; ++stage) {
            for (int stride = (1 << stage); stride > 0; stride /= 2) {
                bool reverse = (lane >> stage) & 2;
                bool is_second = lane & stride;
                T my_val = *val_arr;
                idxT my_idx = *idx_arr;
                
                T other = sycl::select_from_group(sg, my_val, lane ^ stride);
                idxT other_idx = sycl::select_from_group(sg, my_idx, lane ^ stride);

                bool is_better;
                if constexpr (is_stable) {
                    if constexpr (ascending) {
                        is_better = ((my_val > other) ||
                            ((my_val == other) && (my_idx < other_idx))) != (reverse != is_second);
                    } else {
                        is_better = ((my_val > other) ||
                            ((my_val == other) && (my_idx < other_idx))) != (reverse != is_second);
                    }
                } else {
                    is_better = (my_val != other &&
                        (my_val > other) != (reverse != is_second));
                }
                if (is_better) {
                    *val_arr = other;
                    *idx_arr = other_idx;
                }
            }
        }
        BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr, idx_arr, item);
    }
};

template <bool ascending, bool reverse, typename T, typename idxT, bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
    static void merge(T* val_arr, idxT* idx_arr, sycl::nd_item<1> item) {
        int lane = item.get_local_id(0) % WARP_SIZE;
        auto sg = item.get_sub_group();
        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
            bool is_second = lane & stride;
            T val = *val_arr;
            idxT idx = *idx_arr;
            T other = sycl::select_from_group(sg, val, lane ^ stride);
            idxT other_idx = sycl::select_from_group(sg, idx, lane ^ stride);
            
            bool is_better;
            if constexpr (is_stable) {
                if constexpr (ascending) {
                    is_better = ((val > other) ||
                        ((val == other) && (idx < other_idx))) == (reverse != is_second);
                } else {
                    is_better = ((val > other) ||
                        ((val == other) && (idx < other_idx))) == (reverse != is_second);
                }
            } else {
                is_better = (val != other && ((val > other) == (ascending != is_second)));
            }
            if (is_better) {
                *val_arr = other;
                *idx_arr = other_idx;
            }
        }
    }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
    WarpSort(idxT k, T dummy, sycl::nd_item<1> item)
        : lane_(item.get_local_id(0) % WARP_SIZE), k_(k), dummy_(dummy), item_(item) {
        static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));
        for (int i = 0; i < max_arr_len_; ++i) {
            val_arr_[i] = dummy_;
            idx_arr_[i] = 0;
        }
    }

    void load_sorted(const T* in, const idxT* in_idx, idxT start) {
        idxT idx = start + WARP_SIZE - 1 - lane_;
        for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
            if (idx < start + k_) {
                T t = in[idx];
                bool is_better;
                if constexpr (is_stable) {
                    is_better = is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
                } else {
                    is_better = is_better_than<greater>(t, val_arr_[i]);
                }
                if (is_better) {
                    val_arr_[i] = t;
                    idx_arr_[i] = in_idx[idx];
                }
            }
        }
        BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(val_arr_, idx_arr_, item_);
    }

    void dump(T* out, idxT* out_idx) const {
        for (int i = 0; i < max_arr_len_; ++i) {
            idxT out_i = i * WARP_SIZE + lane_;
            if (out_i < k_) {
                out[out_i] = val_arr_[i];
                out_idx[out_i] = idx_arr_[i];
            }
        }
    }

    void dumpIdx(idxT* out_idx) const {
        for (int i = 0; i < max_arr_len_; ++i) {
            idxT out_i = i * WARP_SIZE + lane_;
            if (out_i < k_) {
                out_idx[out_i] = idx_arr_[i];
            }
        }
    }

    idxT get_idx(int i = 0) const {
        return idx_arr_[i];
    }

    T get_val(int i = 0) const { return val_arr_[i]; }

 protected:
    static constexpr int max_arr_len_ = capacity / WARP_SIZE;
    T val_arr_[max_arr_len_];
    idxT idx_arr_[max_arr_len_];
    int const lane_;
    idxT const k_;
    T const dummy_;
    sycl::nd_item<1> item_;
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
    WarpSelect(idxT k, T dummy, sycl::nd_item<1> item)
        : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy, item),
          k_th_(dummy),
          k_th_idx_(0),
          k_th_lane_((k - 1) % WARP_SIZE),
          smem_buf_len_(0) {
                int const warp_id = item.get_local_id(0) / WARP_SIZE;
                T* val_smem_base = *sycl::ext::oneapi::group_local_memory_for_overwrite<T[MaxSupportedExpertCount]>(item.get_group());
                idxT* idx_smem_base = *sycl::ext::oneapi::group_local_memory_for_overwrite<idxT[MaxSupportedExpertCount]>(item.get_group());
                val_smem_ = val_smem_base + warp_id * WARP_SIZE;
                idx_smem_ = idx_smem_base + warp_id * WARP_SIZE;
    }

    void add(T const* in, idxT start, idxT end) {
        idxT const end_for_fullwarp = round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
        for (idxT i = start + this->lane_; i < end_for_fullwarp; i += WARP_SIZE) {
            T val = (i < end) ? in[i] : this->dummy_;
            add(val, i);
        }
    }

    void add(T val, idxT idx) {
        bool do_add;
        if constexpr (is_stable) {
            do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
        } else {
            do_add = is_better_than<greater>(val, k_th_);
        }

        auto subgroup = this->item_.get_sub_group();
        int selected = do_add ? 1 : 0;
        int total_selected =
            sycl::reduce_over_group(subgroup, selected, sycl::plus<int>());
        if (total_selected == 0) {
            return;
        }
        int prefix_selected = sycl::exclusive_scan_over_group(
            subgroup, selected, sycl::plus<int>());

        int pos = smem_buf_len_ + prefix_selected;
        if (do_add && pos < WARP_SIZE) {
            val_smem_[pos] = val;
            idx_smem_[pos] = idx;
            do_add = false;
        }
        smem_buf_len_ += total_selected;
        if (smem_buf_len_ >= WARP_SIZE) {
            sycl::group_barrier(subgroup);
            merge_buf_(val_smem_[this->lane_], idx_smem_[this->lane_]);
            smem_buf_len_ -= WARP_SIZE;
        }
        if (do_add) {
            pos -= WARP_SIZE;
            val_smem_[pos] = val;
            idx_smem_[pos] = idx;
        }
        sycl::group_barrier(subgroup);
    }

    void done() {
        if (smem_buf_len_) {
            T val = (this->lane_ < smem_buf_len_) ? val_smem_[this->lane_] : this->dummy_;
            idxT idx = (this->lane_ < smem_buf_len_) ? idx_smem_[this->lane_] : 0;
            merge_buf_(val, idx);
        }
    }

 private:
    void set_k_th_() {
        auto subgroup = this->item_.get_sub_group();
        k_th_ = sycl::select_from_group(
            subgroup, this->val_arr_[this->max_arr_len_ - 1], k_th_lane_);
        if constexpr (is_stable) {
            k_th_idx_ = sycl::select_from_group(
                subgroup, this->idx_arr_[this->max_arr_len_ - 1],
                k_th_lane_);
        }
    }

    void merge_buf_(T val, idxT idx) {
        BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx, this->item_);
        T& old = this->val_arr_[this->max_arr_len_ - 1];
        bool is_better;
        if constexpr (is_stable) {
            is_better = is_better_than<greater>(val, old, idx, this->idx_arr_[this->max_arr_len_ - 1]);
        } else {
            is_better = is_better_than<greater>(val, old);
        }
        if (is_better) {
            old = val;
            this->idx_arr_[this->max_arr_len_ - 1] = idx;
        }
        BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(this->val_arr_, this->idx_arr_, this->item_);
        set_k_th_();
    }

    T* val_smem_;
    idxT* idx_smem_;
    int smem_buf_len_;
    T k_th_;
    idxT k_th_idx_;
    int const k_th_lane_;
};

}  // namespace warp_topk

namespace reduce_topk {

template <int N_IN, typename T, typename IdxT>
inline void reduceTopK(sycl::sub_group subgroup, T* out_val, IdxT* out_idx,
                       const T* in_vals, const IdxT* in_idxs, T min_val,
                       int topk) {
    constexpr IdxT invalid_idx = std::numeric_limits<IdxT>::max();
    bool selected[N_IN] = {false};

    for (int k = 0; k < topk; ++k) {
        using CT = compute_type_t<T>;
        CT local_best_val = static_cast<CT>(min_val);
        IdxT local_best_idx = invalid_idx;
        int local_best_pos = -1;

        #pragma unroll
        for (int i = 0; i < N_IN; ++i) {
            if (selected[i]) {
                continue;
            }
            T cand_val = in_vals[i];
            IdxT cand_idx = in_idxs[i];
            if ((cand_val > local_best_val) ||
                ((cand_val == local_best_val) && (cand_idx < local_best_idx))) {
                local_best_val = cand_val;
                local_best_idx = cand_idx;
                local_best_pos = i;
            }
        }

        T warp_best_val = sycl::reduce_over_group(
            subgroup, local_best_val, sycl::maximum<CT>());

        IdxT warp_best_idx = invalid_idx;
        if (local_best_pos != -1 && local_best_val == warp_best_val) {
            warp_best_idx = local_best_idx;
        }
        warp_best_idx = sycl::reduce_over_group(
            subgroup, warp_best_idx, sycl::minimum<IdxT>());

        bool found = (warp_best_idx != invalid_idx);
        if (found) {
            int insert_pos = k;
            while (insert_pos > 0 && out_val[insert_pos - 1] == warp_best_val &&
                   out_idx[insert_pos - 1] > warp_best_idx) {
                out_val[insert_pos] = out_val[insert_pos - 1];
                out_idx[insert_pos] = out_idx[insert_pos - 1];
                --insert_pos;
            }
            out_val[insert_pos] = warp_best_val;
            out_idx[insert_pos] = warp_best_idx;
        } else {
            out_val[k] = min_val;
            out_idx[k] = 0;
        }

        if (found && local_best_pos != -1 && local_best_val == warp_best_val &&
            local_best_idx == warp_best_idx) {
            selected[local_best_pos] = true;
        }
    }
}

template <typename T, typename IdxT>
inline void reduceTopK(sycl::sub_group subgroup, T* out_val, IdxT* out_idx,
                       T val, IdxT idx, T min_val, int topk) {
    T in_vals[1] = {val};
    IdxT in_idxs[1] = {idx};
    reduceTopK<1>(subgroup, out_val, out_idx, in_vals, in_idxs, min_val,
                  topk);
}

}  // namespace reduce_topk


template <typename T, typename BiasT, ScoringFunc SF>
SYCL_EXTERNAL inline void topk_with_k2(T* output, T const* input,
                                       BiasT const* bias,
                  sycl::nd_item<1> const& item,
                  int32_t const lane_id,
                  int const num_experts_per_group) {
    T largest = neg_inf<T>();
    T second_largest = neg_inf<T>();
    bool has_bias = (bias != nullptr);
    if (num_experts_per_group > WARP_SIZE) {
        for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
            T value = apply_scoring<SF>(input[i]);
            value = has_bias ? value + sycl_cast<T, BiasT>(bias[i]):value;
            if (value > largest) {
                second_largest = largest;
                largest = value;
            } else if (value > second_largest) {
                second_largest = value;
            }
        }
    } else {
        for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
            T value = apply_scoring<SF>(input[i]);
            value = apply_scoring<SF>(input[i]);
            value = has_bias ? value + sycl_cast<T, BiasT>(bias[i]):value;
            largest = value;
        }
    }

    auto group = item.get_sub_group();
    using CT = compute_type_t<T>;
    CT ct_largest = static_cast<CT>(largest);
    CT ct_max1 = sycl::reduce_over_group(group, ct_largest, sycl::maximum<CT>());
    T max1 = static_cast<T>(ct_max1);
    T max2 = max1;

    bool equal_to_max1 = (max1 == largest);

    int count_max1 = sycl::reduce_over_group(group, equal_to_max1 ? 1 : 0, sycl::plus<int>());

    if (count_max1 == 1) {
        largest = (largest == max1) ? second_largest : largest;
        CT ct_sec = static_cast<CT>(largest);
        CT ct_max2 = sycl::reduce_over_group(group, ct_sec, sycl::maximum<CT>());
        max2 = static_cast<T>(ct_max2);
    }

    if (lane_id == 0) {
        *output = max1 + max2;
    }
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
SYCL_EXTERNAL inline void grouped_topk_fused_kernel(
    T* scores, float* topk_values, IdxT* topk_indices, BiasT const* bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, bool renormalize,
    double routed_scaling_factor, sycl::nd_item<1> item) {
    int32_t token_id = item.get_group(0);
    if (token_id >= num_tokens) return;

    int32_t thread_id = item.get_local_id(0);
    int32_t warp_id = thread_id / WARP_SIZE;
    int32_t lane_id = thread_id % WARP_SIZE;

    int32_t n_group_i32 = static_cast<int32_t>(n_group);
    int32_t topk_group_i32 = static_cast<int32_t>(topk_group);
    int32_t topk_i32 = static_cast<int32_t>(topk);
    int32_t num_experts_i32 = static_cast<int32_t>(num_experts);
    bool has_bias = (bias != nullptr);

    int32_t num_warps = item.get_local_range(0) / WARP_SIZE;
    if (warp_id >= n_group_i32 || num_warps < n_group_i32) return;

    int32_t num_experts_per_group = num_experts_i32 / n_group_i32;

    T* scores_token = scores + static_cast<int64_t>(token_id) * num_experts;

    T* s_group_scores = *sycl::ext::oneapi::group_local_memory_for_overwrite<T[32]>(item.get_group()); // Max n_group_i32 is 32.

    int32_t group_offset = warp_id * num_experts_per_group;
    topk_with_k2<T, BiasT, SF>(s_group_scores + warp_id, // This accesses smem at warp_id.
                               scores_token + group_offset,
                               has_bias ? (bias + group_offset) : nullptr,
                               item, lane_id, num_experts_per_group);

    item.barrier(sycl::access::fence_space::local_space);

    if (warp_id != 0) return;

    topk_values += static_cast<int64_t>(token_id) * topk;
    topk_indices += static_cast<int64_t>(token_id) * topk;

    warp_topk::WarpSelect<WARP_SIZE, true, T, int32_t, true>
        group_sel(static_cast<int32_t>(topk_group_i32), neg_inf<T>(), item);

    T gscore = (lane_id < n_group_i32) ? s_group_scores[lane_id] : neg_inf<T>();
    group_sel.add(gscore, lane_id);
    group_sel.done();

    bool proceed = false;
    if (topk_group_i32 > 0) {
        int kth_lane = topk_group_i32 - 1;
        // Broadcast from lane k_th_lane
        T kth_val = sycl::select_from_group(item.get_sub_group(), group_sel.get_val(0), kth_lane);
        proceed = (kth_val != neg_inf<T>());
    }

    if (!proceed) {
        for (int i = lane_id; i < topk_i32; i += WARP_SIZE) {
            topk_indices[i] = static_cast<IdxT>(i);
            topk_values[i] = 1.0f / static_cast<float>(topk_i32);
        }
        return;
    }

    warp_topk::WarpSelect<WARP_SIZE, true, T, int32_t, true>
        expert_sel(static_cast<int32_t>(topk_i32), neg_inf<T>(), item);

    int32_t sel_gid_lane = (lane_id < topk_group_i32) ? group_sel.get_idx(0) : 0;
    int32_t total_candidates = topk_group_i32 * num_experts_per_group;
    T cand = neg_inf<T>();
    int32_t cand_idx = 0;
    for (int32_t g = 0; g < topk_group_i32; ++g) {
        int32_t gid = sycl::select_from_group(item.get_sub_group(), sel_gid_lane, g);
        int32_t offset = gid * num_experts_per_group;
        int32_t align_num_experts_per_group =
            warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);
        for (int32_t i = lane_id; i < align_num_experts_per_group; i += WARP_SIZE) {
            T cand = neg_inf<T>();
            int32_t idx = 0;
            if (i < num_experts_per_group) {
                idx = offset + i;
                T input = scores_token[idx];
                if (is_finite(input)) {
                    T score = apply_scoring<SF>(input);
                    cand = score;
                    if (has_bias) {
                        cand = cand + sycl_cast<T, BiasT>(bias[idx]);
                    }
                }
            }
            expert_sel.add(cand, idx);
        }
    }
    expert_sel.done();


    float lane_unbiased = 0.0f;
    IdxT lane_idx = 0;
    if (lane_id < topk_i32) {
        lane_idx = static_cast<IdxT>(expert_sel.get_idx(0));
        T in = scores_token[static_cast<int32_t>(lane_idx)];
        lane_unbiased = sycl_cast<float, T>(apply_scoring<SF>(in));
    }

    float topk_sum = 1e-20f;
    if (renormalize) {
        auto group = item.get_sub_group();
        topk_sum += sycl::reduce_over_group(group, lane_unbiased, sycl::plus<float>());
    }

    float scale = static_cast<float>(routed_scaling_factor);
    if (renormalize) {
        scale /= topk_sum;
    }

    if (lane_id < topk_i32) {
        topk_indices[lane_id] = lane_idx;
        topk_values[lane_id] = lane_unbiased * scale;
    }
}


template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, bool UseGroups,
          int MaxNumTopExperts = DefaultMaxNumTopExperts>
SYCL_EXTERNAL inline void grouped_topk_fused_small_expert_count_kernel(
    T* scores, float* topkValues, IdxT* topkIndices, BiasT const* routingBias,
    int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup,
    int64_t const topk, int64_t const numExperts,
    int64_t const numExpertsPerGroup, bool const renormalize,
    double const routedScalingFactor, sycl::nd_item<1> item) {

    constexpr int NumWarps = MaxNumExperts / WARP_SIZE;
    constexpr float invalidScoreFloat = -std::numeric_limits<float>::infinity();

    int threadIdx = item.get_local_id(0);
    int blockIdx = item.get_group(0);
    if constexpr (UseGroups){
        if (blockIdx >= numTokens) return;
    }
    int localSize = item.get_local_range(0);
    bool has_bias = (routingBias != nullptr);

    int laneIdx = threadIdx % WARP_SIZE;
    int warpIdx = threadIdx / WARP_SIZE;
    

    topkValues += blockIdx * topk;
    topkIndices += blockIdx * topk;

    if constexpr (UseGroups) {
        auto subgroup = item.get_sub_group();
        T* scoresToken = scores + static_cast<int64_t>(blockIdx) * numExperts;
        T selectedGroupScores[WARP_SIZE];
        int32_t selectedGroupIdx[WARP_SIZE];

        T groupScore = neg_inf<T>();
        if (laneIdx < numGroup) {
            int32_t groupOffset = laneIdx * numExpertsPerGroup;
            T largest = neg_inf<T>();
            T secondLargest = neg_inf<T>();

            for (int32_t i = 0; i < numExpertsPerGroup; ++i) {
                T value = apply_scoring<SF>(scoresToken[groupOffset + i]);
                if (has_bias) {
                    value = value + sycl_cast<T, BiasT>(routingBias[groupOffset + i]);
                }
                if (value > largest) {
                    secondLargest = largest;
                    largest = value;
                } else if (value > secondLargest) {
                    secondLargest = value;
                }
            }
            groupScore = has_bias ? largest + secondLargest : largest;
        }

        reduce_topk::reduceTopK(
            subgroup, selectedGroupScores, selectedGroupIdx,
            groupScore, laneIdx, neg_inf<T>(), static_cast<int>(topkGroup));

        bool proceed = false;
        if (topkGroup > 0) {
            proceed = (selectedGroupScores[topkGroup - 1] != neg_inf<T>());
        }

        if (!proceed) {
            for (int i = laneIdx; i < topk; i += WARP_SIZE) {
                topkIndices[i] = static_cast<IdxT>(i);
                topkValues[i] = 1.0f / static_cast<float>(topk);
            }
            return;
        }

        constexpr int MaxExpertCandidatesPerLane = NumDeepseekExperts / WARP_SIZE;
        T localCandidateScores[MaxExpertCandidatesPerLane];
        IdxT localCandidateIdx[MaxExpertCandidatesPerLane];
        T selectedExpertScores[DefaultMaxNumTopExperts];
        IdxT selectedExpertIdx[DefaultMaxNumTopExperts];

        for (int i = 0; i < MaxExpertCandidatesPerLane; ++i) {
            localCandidateScores[i] = neg_inf<T>();
            localCandidateIdx[i] = 0;
        }

        int32_t totalCandidates = topkGroup * numExpertsPerGroup;
        for (int32_t candidate = laneIdx; candidate < totalCandidates;
             candidate += WARP_SIZE) {
            int32_t localSlot = candidate / WARP_SIZE;
            int32_t selectedGroup = candidate / numExpertsPerGroup;
            int32_t expertInGroup = candidate % numExpertsPerGroup;
            int32_t gid = selectedGroupIdx[selectedGroup];
            int32_t idx = gid * numExpertsPerGroup + expertInGroup;
            T candidateScore = neg_inf<T>();

            T input = scoresToken[idx];
            if (is_finite(input)) {
                T score = apply_scoring<SF>(input);
                candidateScore = score;
                if (has_bias) {
                    candidateScore = candidateScore + sycl_cast<T, BiasT>(routingBias[idx]);
                }
            }

            localCandidateScores[localSlot] = candidateScore;
            localCandidateIdx[localSlot] = static_cast<IdxT>(idx);
        }

        reduce_topk::reduceTopK<MaxExpertCandidatesPerLane>(
            subgroup, selectedExpertScores, selectedExpertIdx,
            localCandidateScores, localCandidateIdx, neg_inf<T>(), static_cast<int>(topk));

        for (int i = 1; i < topk; ++i) {
            T score = selectedExpertScores[i];
            IdxT idx = selectedExpertIdx[i];
            int j = i;
            while (j > 0 &&
                   ((selectedExpertScores[j - 1] < score) ||
                    ((selectedExpertScores[j - 1] == score) &&
                     (selectedExpertIdx[j - 1] > idx)))) {
                selectedExpertScores[j] = selectedExpertScores[j - 1];
                selectedExpertIdx[j] = selectedExpertIdx[j - 1];
                --j;
            }
            selectedExpertScores[j] = score;
            selectedExpertIdx[j] = idx;
        }

        float laneUnbiased = 0.0f;
        IdxT laneIdxOut = 0;
        if (laneIdx < topk) {
            laneIdxOut = selectedExpertIdx[laneIdx];
            T in = scoresToken[static_cast<int32_t>(laneIdxOut)];
            laneUnbiased = sycl_cast<float, T>(apply_scoring<SF>(in));
        }

        float scale = static_cast<float>(routedScalingFactor);
        if (renormalize) {
            float topkSum = 1e-20f;
            topkSum += sycl::reduce_over_group(
                subgroup, laneUnbiased,sycl::plus<float>());
            scale /= topkSum;
        }

        if (laneIdx < topk) {
            topkIndices[laneIdx] = laneIdxOut;
            topkValues[laneIdx] = laneUnbiased * scale;
        }
        return;
    } else {

    float* smemScoreSigmoid = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MaxNumExperts]>(item.get_group());
    float* smemScoreBias = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[MaxNumExperts]>(item.get_group());
    float topScores[MaxNumTopExperts] = {invalidScoreFloat};
    int32_t topExperts[MaxNumTopExperts] = {0};
    float expertScoreGroup[MaxNumTopGroups] = {invalidScoreFloat};
    int32_t expertIdxGroup[MaxNumTopGroups] = {0};
    auto group = item.get_sub_group();

    for (int expert = threadIdx; expert < numExperts; expert += localSize) {
        int64_t scoreIdx = int64_t{blockIdx} * int64_t{numExperts} + expert;
        float score = sycl_cast<float, T>(scores[scoreIdx]);
        float scoreSigmoid = apply_scoring<SF>(score);
        smemScoreSigmoid[expert] = scoreSigmoid;
        smemScoreBias[expert] = has_bias
            ? (scoreSigmoid + sycl_cast<float, BiasT>(routingBias[expert]))
            : scoreSigmoid;
    }

    if constexpr (MaxNumExperts > MaxNumExpertsUnit) {
        constexpr int NumExpertWarps = (MaxNumExperts - 1) / MaxNumExpertsUnit + 1;
        constexpr int NumInterTopK = NumExpertWarps * MaxNumTopExperts;
        float* smemInterTopScores = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[NumInterTopK]>(item.get_group());
        int32_t* smemInterTopExperts = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[NumInterTopK]>(item.get_group());

        if (warpIdx < NumExpertWarps) {
            int offset = warpIdx * WARP_SIZE * MaxNumTopGroups;

            for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
                int expertIdx = ii * WARP_SIZE + laneIdx;
                expertIdxGroup[ii] = offset + expertIdx;
                expertScoreGroup[ii] = (offset + expertIdx < numExperts)
                                           ? smemScoreBias[offset + expertIdx]
                                           : invalidScoreFloat;
            }
            reduce_topk::reduceTopK<MaxNumTopGroups>(
                group, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                invalidScoreFloat, static_cast<int>(topk));

            if (laneIdx < MaxNumTopExperts) {
                if (laneIdx < topk) {
                    smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] = topScores[laneIdx];
                    smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] = topExperts[laneIdx];
                } else {
                    smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] = invalidScoreFloat;
                    smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] = MaxNumExperts - 1;
                }
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (warpIdx == 0) {
            constexpr int NumInterTopKPerThread = (NumInterTopK - 1) / WARP_SIZE + 1;
            float intermediateScore[NumInterTopKPerThread];
            int32_t intermediateExpert[NumInterTopKPerThread];

            for (int i = laneIdx; i < NumInterTopKPerThread * WARP_SIZE; i += WARP_SIZE) {
                int ii = i / WARP_SIZE;
                if (i < NumInterTopK) {
                    intermediateScore[ii] = smemInterTopScores[i];
                    intermediateExpert[ii] = smemInterTopExperts[i];
                } else {
                    intermediateScore[ii] = invalidScoreFloat;
                    intermediateExpert[ii] = MaxNumExperts - 1;
                }
            }

            reduce_topk::reduceTopK<NumInterTopKPerThread>(
                group, topScores, topExperts, intermediateScore, intermediateExpert,
                invalidScoreFloat, static_cast<int>(topk));
        }
    } else {
        if (warpIdx == 0) {
            for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
                int32_t expertIdx = ii * WARP_SIZE + laneIdx;
                expertIdxGroup[ii] = expertIdx;
                expertScoreGroup[ii] = (expertIdx < numExperts)
                                           ? smemScoreBias[expertIdx]
                                           : invalidScoreFloat;
            }
            reduce_topk::reduceTopK<MaxNumTopGroups>(
                group, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                invalidScoreFloat, static_cast<int>(topk));
        }
    }

    if (warpIdx == 0) {
        int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
        float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
        float finalScore = static_cast<float>(scoreNorm * routedScalingFactor);
        float topk_sum = 1e-20f;
        if (renormalize) {
            topk_sum += sycl::reduce_over_group(group, scoreNorm,sycl::plus<float>());
            finalScore /= topk_sum;
        }
        if (laneIdx < topk) {
            topkIndices[laneIdx] = finalScore;
            topkValues[laneIdx] = expertIdx;
        }
    }
    } // end if constexpr (!UseGroups)
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
void invokeNoAuxTc(T* scores, float* topk_values, IdxT* topk_indices,
                   BiasT const* bias, int64_t const num_tokens,
                   int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk,
                   bool const renormalize, double const routed_scaling_factor,
                   bool enable_pdl = false, sycl::queue queue = sycl::queue()) {
    int64_t experts_per_group = num_experts / n_group;
    bool is_single_group =
        (n_group == 1) && (topk_group == 1) &&
        (num_experts <= MaxSupportedExpertCount) &&
        (topk <= DefaultMaxNumTopExperts || topk == MaxSupportedTopExperts);

    bool is_multi_group =
        (n_group > 1) && (num_experts <= NumDeepseekExperts) &&
        (experts_per_group <= WARP_SIZE) &&
        (experts_per_group * topk_group <= MaxNumExpertsUnit) &&
        (topk <= DefaultMaxNumTopExperts) && (topk_group <= MaxNumTopGroups);

    if (is_single_group || is_multi_group) {
        #define LAUNCH_SMALL_KERNEL(MAX_EXPERTS, USE_GROUPS, MAX_TOP_EXPERTS, NUM_THREADS) \
        do { \
            size_t local_size = static_cast<size_t>(NUM_THREADS); \
            size_t global_size = static_cast<size_t>(num_tokens) * local_size; \
            queue.submit([&](sycl::handler& cgh) { \
                cgh.parallel_for<VllmGroupedTopKFusedSmallExpertCountKernel<T, BiasT, IdxT, SF, MAX_EXPERTS, USE_GROUPS, MAX_TOP_EXPERTS>>( \
                    sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)), \
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] { \
                        grouped_topk_fused_small_expert_count_kernel<T, BiasT, IdxT, SF, MAX_EXPERTS, USE_GROUPS, MAX_TOP_EXPERTS>( \
                            scores, topk_values, topk_indices, bias, \
                            num_tokens, n_group, topk_group, topk, num_experts, \
                            experts_per_group, renormalize, routed_scaling_factor, item); \
                    }); \
            }); \
        } while (0)

        if (is_single_group) {
            if (num_experts == NumNemotronExperts && n_group == 1 &&
                topk == MaxSupportedTopExperts) {
                LAUNCH_SMALL_KERNEL(NumNemotronExperts, false,
                                    MaxSupportedTopExperts,
                                    ((NumNemotronExperts + MaxNumExpertsUnit - 1) /
                                     MaxNumExpertsUnit) * WARP_SIZE);
            } else if (num_experts > NumKimiK2Experts &&
                       num_experts <= MaxSupportedExpertCount) {
                LAUNCH_SMALL_KERNEL(MaxSupportedExpertCount, false,
                                    DefaultMaxNumTopExperts,
                                    ((MaxSupportedExpertCount + MaxNumExpertsUnit - 1) /
                                     MaxNumExpertsUnit) * WARP_SIZE);
            } else if (num_experts > MaxNumExpertsUnit &&
                       num_experts <= NumKimiK2Experts) {
                LAUNCH_SMALL_KERNEL(NumKimiK2Experts, false,
                                    DefaultMaxNumTopExperts,
                                    ((NumKimiK2Experts + MaxNumExpertsUnit - 1) /
                                     MaxNumExpertsUnit) * WARP_SIZE);
            } else {
                LAUNCH_SMALL_KERNEL(MaxNumExpertsUnit, false,
                                    DefaultMaxNumTopExperts,
                                    WARP_SIZE);
            }
        } else {
            LAUNCH_SMALL_KERNEL(NumDeepseekExperts, true,
                                DefaultMaxNumTopExperts,
                                WARP_SIZE);
        }

        #undef LAUNCH_SMALL_KERNEL
    } else {
        size_t local_size = static_cast<size_t>(n_group) * WARP_SIZE;
        size_t global_size = static_cast<size_t>(num_tokens) * local_size;;

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<VllmGroupedTopKFusedKernel<T, BiasT, IdxT, SF>>(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    grouped_topk_fused_kernel<T, BiasT, IdxT, SF>(
                        scores, topk_values, topk_indices, bias,
                        num_tokens, num_experts, n_group, topk_group, topk,
                        renormalize, routed_scaling_factor, item);
                });
        });
    }
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT, SF)                             \
  template void invokeNoAuxTc<T, BiasT, IdxT, SF>(                           \
      T * scores, float* topk_values, IdxT* topk_indices, BiasT const* bias, \
      int64_t const num_tokens, int64_t const num_experts,                   \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,   \
      bool const renormalize, double const routed_scaling_factor,            \
      bool enable_pdl, sycl::queue queue);

INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::half, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, int32_t, SCORING_SIGMOID);
INSTANTIATE_NOAUX_TC(float, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(float, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::half, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::half, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, float, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, sycl::half, int32_t, SCORING_NONE);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, int32_t, SCORING_NONE);
}  // end namespace moe
}  // namespace vllm

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    c10::optional<torch::Tensor> const& bias, int64_t scoring_func = 0) {
    auto data_type = scores.scalar_type();
    bool has_bias = bias.has_value() && bias->defined();
    auto bias_type = has_bias ? bias->scalar_type() : torch::kFloat32;
    auto input_size = scores.sizes();
    int64_t num_tokens = input_size[0];
    int64_t num_experts = input_size[1];
    
    TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
    TORCH_CHECK(n_group > 0, "n_group must be positive");
    TORCH_CHECK(topk > 0, "topk must be positive");
    TORCH_CHECK(topk_group > 0, "topk_group must be positive");
    TORCH_CHECK(topk_group <= n_group, "topk_group must be <= n_group");
    TORCH_CHECK(num_experts % n_group == 0,
                "num_experts should be divisible by n_group");
    TORCH_CHECK(n_group <= 32,
                "n_group should be smaller than or equal to 32 for now");
    TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
    TORCH_CHECK(topk <= topk_group * (num_experts / n_group),
                "topk must be <= topk_group * (num_experts / n_group)");
    TORCH_CHECK(scoring_func == vllm::moe::SCORING_NONE ||
                    scoring_func == vllm::moe::SCORING_SIGMOID,
                "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  // Always output float32 for topk_values (eliminates Python-side conversion)
    torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kFloat32).device(scores.device()));
    torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(scores.device()));

    auto device_idx = scores.device().index();
    auto stream = c10::xpu::getCurrentXPUStream(device_idx).queue();
    auto const sf = static_cast<vllm::moe::ScoringFunc>(scoring_func);

#define LAUNCH_KERNEL_SF(T, BiasT, IdxT)                                      \
  do {                                                                        \
    switch (sf) {                                                             \
      case vllm::moe::SCORING_NONE:                                           \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_NONE>(    \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                  \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            (has_bias ? reinterpret_cast<BiasT const*>(bias->data_ptr()) : nullptr), num_tokens,      \
            num_experts, n_group, topk_group, topk, renormalize,              \
            routed_scaling_factor, false, stream);                            \
        break;                                                                \
      case vllm::moe::SCORING_SIGMOID:                                        \
        vllm::moe::invokeNoAuxTc<T, BiasT, IdxT, vllm::moe::SCORING_SIGMOID>( \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                  \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),         \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),         \
            (has_bias ? reinterpret_cast<BiasT const*>(bias->data_ptr()) : nullptr), num_tokens,      \
            num_experts, n_group, topk_group, topk, renormalize,              \
            routed_scaling_factor, false, stream);                            \
        break;                                                                \
      default:                                                                \
        throw std::invalid_argument("Unsupported scoring_func");              \
        break;                                                                \
    }                                                                         \
  } while (0)

#define LAUNCH_KERNEL(T, IdxT)                                             \
  do{                                                                      \
        switch (bias_type) {                                               \
        case torch::kFloat16:                                              \
            LAUNCH_KERNEL_SF(T, sycl::half, IdxT);                         \
            break;                                                         \
        case torch::kFloat32:                                              \
            LAUNCH_KERNEL_SF(T, float, IdxT);                              \
            break;                                                         \
        case torch::kBFloat16:                                             \
            LAUNCH_KERNEL_SF(T, sycl::ext::oneapi::bfloat16, IdxT);                              \
            break;                                                         \
        default:                                                           \
            throw std::invalid_argument(                                   \
                "Invalid bias dtype, only supports float16, float32, and " \
                "bfloat16");                                               \
            break;                                                         \
        }                                                                  \
    }                                                                      \
   while (0)


  switch (data_type) {
    case torch::kFloat16:
      LAUNCH_KERNEL(sycl::half, int32_t);
      break;
    case torch::kFloat32:
      LAUNCH_KERNEL(float, int32_t);
      break;
    case torch::kBFloat16:
      LAUNCH_KERNEL(sycl::ext::oneapi::bfloat16, int32_t);
      break;
    default:
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_KERNEL
#undef LAUNCH_KERNEL_SF
  return {topk_values, topk_indices};
}