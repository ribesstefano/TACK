import setuptools

def read_requirements():
    # Read requirements from file
    with open("requirements.txt") as f:
        return f.read().splitlines()

setuptools.setup(
    name="tackai",
    version="1.0.0",
    author="Stefano Ribes, Nils Dunlop",
    url="https://github.com/ribesstefano/STAEDA",
    author_email="ribes@chalmers.se",
    description="TACK: A statistical evaluation of degradation activity on a novel TArgeting Chimeras Knowledge dataset",
    long_description=open("README.md").read(),
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
