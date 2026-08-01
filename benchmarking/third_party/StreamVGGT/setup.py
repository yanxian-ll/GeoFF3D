from setuptools import find_namespace_packages, setup


setup(
    name="streamvggt",
    version="0.0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["visual_util"],
)
