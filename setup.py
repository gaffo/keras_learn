from setuptools import setup

# on windows install the following in your venv
# via pip:
# cryptography jsonpickle pypiwin32

setup(name='keras-learn',
      version='1.0',
      description='keras-lern',
      author='keras-learn',
      author_email='keras-learn',
      packages=['keras-learn'],
      install_requires=['numpy','keras', 'tensorflow'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'])