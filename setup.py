from setuptools import setup, find_packages

setup(
    name="involution",
    version="0.1.0",
    url="https://github.com/shikishima-TasakiLab/involution",
    license="MIT License",
    author="Junya Shikishima",
    author_email="160442065@ccalumni.meijo-u.ac.jp",
    description="PyTorch Involution",
    packages=find_packages(),
    install_requires=["torch>=1.7.0", "cupy"]
)
