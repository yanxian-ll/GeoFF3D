from setuptools import setup, find_packages

setup(
    name='vggt_slam',
    version='0.1.0',
    description='A feedforward SLAM system optimized on the SL(4) manifold.',
    packages=find_packages(include=['evals', 'evals.*', 'vggt_slam', 'vggt_slam.*']),
)

