#pragma once

#include <sycl/sycl.hpp>

#if __has_include(<c10/xpu/XPUStream.h>)
#include <c10/xpu/XPUStream.h>
#endif

namespace vllm {
namespace xpu {

inline sycl::queue& vllmGetQueue() {
#if __has_include(<c10/xpu/XPUStream.h>)
  return c10::xpu::getCurrentXPUStream().queue();
#else
  static sycl::queue queue{sycl::default_selector_v};
  return queue;
#endif
}

}  // namespace xpu
}  // namespace vllm
