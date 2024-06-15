# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
from setuptools import find_packages, setup
import os
import os.path as osp
ROOT = os.path.dirname(os.path.abspath(__file__))

_ext_src_root = "_ops"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
exec(open(osp.join("_version.py")).read())
requirements = ["torch>=1.4"]
# os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.7;8.9;9.0"
setup(
    name='pointnet22',
    ext_modules=[
        CUDAExtension(
            name='pointnet22._ext',
            version=__version__,
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/{}/include".format(ROOT, _ext_src_root))],
            },
            include_dirs=[osp.join(ROOT, _ext_src_root, "include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


