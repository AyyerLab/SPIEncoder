import setuptools

with open('README.md', 'r') as fptr:
    long_desc = fptr.read()

setuptools.setup(
    name='spiencoder',
    version='0.0.0',
    author='CNI Group at MPSD',
    author_email='kartik.ayyer@mpsd.mpg.de',
    description='NN-encoders for heterogeneity classification in SPI',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    #packages=setuptools.find_packages(),
    packages=['SPIEncoder'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy>=1.0.0',
        'scikit-learn',
        'h5py',
        'torch',
        'cupy',
    ],
)
