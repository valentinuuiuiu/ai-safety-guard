from setuptools import setup, find_packages

setup(
    name="ai-safety-guard",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pytest>=6.0.0",
        "requests>=2.25.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0"
    ],
    author="AI Safety Team",
    author_email="ai-safety@example.com",
    description="A content safety classifier to detect potentially harmful text prompts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-safety-guard",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Security",
        "Topic :: Text Processing :: Linguistic"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "ai-safety-guard-api=run_api:main",
            "ai-safety-guard-train=train_model:main",
        ],
    },
)