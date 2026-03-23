import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

sources = [str(ROOT / "sycl_fused_grouped_topk" / "op.sycl")]

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17"]
    sycl_args = ["/O2", "/std:c++17"]
else:
    cxx_args = ["-O3", "-std=c++17"]
    sycl_args = ["-O3"]

ext_modules = [
    SyclExtension(
        name="sycl_fused_grouped_topk._C",
        sources=sources,
        include_dirs=[
            str(ROOT / "shims" / "csrc" / "moe"),
            str(ROOT),
        ],
        extra_compile_args={"cxx": cxx_args, "sycl": sycl_args},
    )
]

setup(
    name="sycl_fused_grouped_topk",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)
