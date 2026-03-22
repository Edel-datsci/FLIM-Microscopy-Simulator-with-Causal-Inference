from setuptools import setup, find_packages

setup(
    name="ed_mrdt",
    version="1.1.0",
    description="Event-Driven Monte Carlo Reversible Dynamics Engine for FLIM-FRET Simulation",
    author="Edel-Cunill",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.56.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pillow>=9.0.0"
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "ed_mrdt": ["examples/*.yaml", "examples/*.py"]
    },
)
