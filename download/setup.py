from setuptools import setup


setup(
    name="wandb-utils",
    version=0.1,
    description="Utilities for interacting with W&B and mlflow",
    zip_safe=False,  # avoid eggs for handling of package data
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "mlflow",
        "wandb"
    ]
)
