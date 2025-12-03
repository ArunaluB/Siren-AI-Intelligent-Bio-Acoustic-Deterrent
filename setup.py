"""
Siren AI v2 - Setup Configuration
==================================

Bio-Inspired Intelligent Acoustic Deterrent System
for Human-Elephant Conflict Mitigation

Author: Bamunusinghe S.A.N. (IT22515612)
Institution: SLIIT Faculty of Computing
License: MIT
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    # Package Metadata
    name="siren-ai-v2",
    version="2.0.0",
    description="Bio-Inspired Intelligent Acoustic Deterrent System for Human-Elephant Conflict",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author Information
    author="Bamunusinghe S.A.N.",
    author_email="it22515612@my.sliit.lk",
    
    # URLs
    url="https://github.com/arunalub/Siren-AI-Intelligent-Bio-Acoustic-Deterrent",

    
    # License
    license="MIT",
    
    # Package Discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python Version
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional Dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "hardware": [
            "pyserial>=3.5",
            "esptool>=3.3",
        ],
    },
    
    # Entry Points (CLI commands)
    entry_points={
        "console_scripts": [
            "siren-ai-train=main:main",
            "siren-ai-test=tests.run_all_tests:main",
        ],
    },
    
    # Package Data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    
    # Classifiers
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Embedded Systems",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python Versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating Systems
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords
    keywords=[
        "reinforcement-learning",
        "wildlife-conservation",
        "human-elephant-conflict",
        "edge-ai",
        "iot",
        "acoustic-deterrent",
        "sarsa",
        "esp32",
        "lora",
    ],
    
    # Zip Safety
    zip_safe=False,
)
