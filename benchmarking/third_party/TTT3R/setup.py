from setuptools import find_namespace_packages, setup


setup(
    name="ttt3r",
    version="0.0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["roma"],
)
