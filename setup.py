from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-risk-analytics",
    version="0.1.0",
    author="Kishore Umaprasad",
    author_email="",
    description="End-to-End Credit Risk Analytics Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ukishore33/End-to-End-Credit-Risk-Analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.0",
        ],
        "dashboard": [
            "streamlit>=1.24.0",
            "plotly>=5.15.0",
        ],
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=3.3.0",
            "shap>=0.42.0",
            "lime>=0.2.0",
        ],
    },
)
