from setuptools import setup, find_packages

setup(
    name="train-transformer-asr",
    version="0.1.0",
    description="Train an ASR model from scratch for a low resource language, comparing transformer decoding vs CTC",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "transformers>=4.35.0",
        "jiwer>=3.0.3",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "tensorboard>=2.14.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "tqdm>=4.66.1",
        "boto3>=1.28.0",
        "azure-ai-ml>=1.11.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
    ],
)
