from setuptools import setup, find_packages

setup(
    name="model_hub",
    version="0.1.0",
    description="A hub for managing machine learning models.",
    author="ksuaning-au",
    author_email="ksuaning@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "openai",
        "google-genai",
    ],
)