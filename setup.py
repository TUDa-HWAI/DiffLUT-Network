import os

from setuptools import setup


ext_modules = []
cmdclass = {}

# DiffLUT-Net itself uses PyTorch operations and does not require a compiled
# extension. The legacy accelerated layers can be enabled explicitly on
# systems whose CUDA Toolkit matches the CUDA version used by PyTorch.
if os.environ.get('DIFFLUT_BUILD_CUDA') == '1':
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules.append(
        CUDAExtension(
            'difflogic_cuda',
            [
                'difflogic/cuda/difflogic.cpp',
                'difflogic/cuda/difflogic_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': ['-std=c++17', '-lineinfo'],
            },
        )
    )
    cmdclass['build_ext'] = BuildExtension

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='difflut-net',
    version='0.1.0',
    description='Differentiable six-input lookup-table networks',
    author='Jiaqi Ye',
    author_email='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    package_dir={'difflogic': 'difflogic'},
    packages=['difflogic', 'optimizers'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'numpy',
    ],
)
