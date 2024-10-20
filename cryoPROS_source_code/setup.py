import os
from Cython.Build import cythonize
from setuptools import setup, Extension

# def version():
#   initPath = os.path.abspath(os.path.join(__file__, "..", "cryoPROS", "__init__.py"))
#   with open(initPath) as f:
#     version = f.read().strip().split('"')[-2]
#   return version
      
# def readme():
#   readmePath = os.path.abspath(os.path.join(__file__, "..", "cryoPROS", "README.md"))
#   try:
#     with open(readmePath) as f:
#       return f.read()
#   except UnicodeDecodeError:
#     try:
#       with open(readmePath, 'r', encoding='utf-8') as f:
#         return f.read()
#     except Exception as e:
#       return "Description not available due to unexpected error: "+str(e)
    
install_requires=[
    'torch',
    'torchvision',
    'mrcfile>=1.3',
    'scipy>=1.6.2',
    'tqdm>=4.59',
    'argparse>=1.4',
    'numpy>=1.21.5',
    'pandas>=1.3.2',
    'opencv-python',
    'matplotlib',
],

# parse_module
# parse_ctf_star_mod = Extension("cryoPROS.parse_ctf_star", ["cryoPROS/parse_ctf_star.py"])
# parse_pose_star_mod = Extension("cryoPROS.parse_pose_star", ["cryoPROS/parse_pose_star.py"])

# gen_mask module
genmask_mod = Extension("cryoPROS.gen_mask", ["cryoPROS/gen_mask.py"])

# main_train module
train_mod = Extension("cryoPROS.main_train", ["cryoPROS/main_train.py"])
train_mp_mod = Extension("cryoPROS.main_train_mp", ["cryoPROS/main_train_mp.py"])

# model_mp/network_mp module
model_mod = Extension("cryoPROS.models.model_hvae", ["cryoPROS/models/model_hvae.py"])
network_mod = Extension("cryoPROS.models.network_hvae", ["cryoPROS/models/network_hvae.py"])
model_mp_mod = Extension("cryoPROS.models.model_mp", ["cryoPROS/models/model_mp.py"])
network_mp_mod = Extension("cryoPROS.models.network_mp", ["cryoPROS/models/network_mp.py"])

# gen_unipose module
gen_unipose_mod = Extension("cryoPROS.gen_unipose", ["cryoPROS/gen_unipose.py"])

# main_generate_stack module
generate_mod = Extension("cryoPROS.main_generate_stack", ["cryoPROS/main_generate_stack.py"])

ext_modules = [generate_mod, network_mod, train_mod, model_mod, genmask_mod, train_mp_mod, model_mp_mod, network_mp_mod, gen_unipose_mod]

# extra compile/link args
for ext_module in ext_modules:
    ext_module.extra_compile_args = ['-std=c99']
    # ext_module.extra_link_args = ['-lstdc++', '-static-libstdc++']

ext_modules_cythonized = cythonize(ext_modules)

setup(name='cryoPROS',
      version="1.0",
      description='Addressing preferred orientation in single-particle cryo-EM through AI-generated auxiliary particles.',
      # long_description=readme(),
      # long_description_content_type="text/markdown",
      keywords='preferred orientation in single-particle cryo-EM',
      url='https://github.com/mxhulab/cryoPROS',
      author='Mingxu Hu',
      author_email='humingxu@smart.org.cn',
      license='Reserved',
      packages=['cryoPROS.data', 'cryoPROS.utils', 'cryoPROS.options'],
      package_data={
        'cryoPROS': ['cryoPROS/cryoPROS.pth'], 
        'cryoPROS.options': ["*.json"], 
      },
      install_requires=install_requires,
      ext_modules=ext_modules_cythonized,
      entry_points={
        'console_scripts': [
          'cryopros-gen-mask=cryoPROS.gen_mask:main',
          'cryopros-recondismic=cryoPROS.main_train_mp:main',
          'cryopros-train=cryoPROS.main_train:main',
          'cryopros-uniform-pose=cryoPROS.gen_unipose:main',
          'cryopros-generate=cryoPROS.main_generate_stack:main'
        ],
      },
      include_package_data=False,
      zip_safe=False)
