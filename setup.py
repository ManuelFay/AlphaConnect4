from setuptools import setup, find_packages

setup(
    name="AlphaConnect4",
    version="DEV",
    description="Connect 4 engines",
    author="manu",
    author_email='manuel.fay@gmail.com',
    packages=find_packages(include=["alphaconnect4", "alphaconnect4.*"]),
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "tqdm",
    ],
    python_requires=">=3.6,<4.0",
)
