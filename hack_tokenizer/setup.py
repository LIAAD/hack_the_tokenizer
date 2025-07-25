from setuptools import setup, find_packages

setup(
    name="hack_tokenizer",  # Replace with your package name
    version="0.1.0",
    author="Duarte Pinto",
    author_email="duarte.pinto7337@gmail.com",
    description="A package for hacking and evaluating tokenizers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=['hack_tokenizer'],
    package_dir={'hack_tokenizer': 'src'},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        # Add other dependencies from your project
    ],
)