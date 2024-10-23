from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='drop-the-bassifier',
    version='0.1.0',
    description='Genre classification machine learning application',
    author='Hillel Gersten, Kyle Schoenhardt, Johnathan Sanchez',
    url='https://github.com/kyleryxn/drop-the-bassifier',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,  # Read requirements dynamically from requirements.txt
    entry_points={
        'console_scripts': [
            'runserver=drop_the_bassifier.app:main',  # Replace with the actual path to your app
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Framework :: Flask',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
