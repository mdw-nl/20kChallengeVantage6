from setuptools import setup, find_packages


setup(
    name="my-flower-algorithm",
    version="0.1",
    description="ADMM-based federated logistic regression",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "vantage6-client",
        "vantage6-tools",
    ],
    python_requires=">=3.9",
)
