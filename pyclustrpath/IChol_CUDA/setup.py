from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusparse_extension',
    ext_modules=[
        CUDAExtension(name='cusparse_extension',
                      sources=['cusparse_csr_extension.cu'],
                      libraries=['cusparse'],  # add cusparse library
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


# python setup.py install
