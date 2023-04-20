import setuptools

setuptools.setup(
    name="ltsm",
    version='1.0.0',
    author="Data Lab",
    author_email="daochen.zha@rice.edu",
    description="Large Time Sereis Model",
    url="XXXX",
    keywords=["Time Series"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
