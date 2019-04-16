# Once in the pyrobolearn folder, type one of the following command: 
# $ pip install -e .
# $ pip install --user -e .
# $ python setup.py install
# $ python setup.py install --home=<dir>

from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

# get description from readme file
with open('README.md', 'r') as f:
    long_description = f.read()

# get the required packages
install_requires = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_requires]

# setup
setup(
    name='pyrobolearn',
    version='0.1.0',
    description='A Python framework for roboticists and machine learning practitioners',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author='Brian Delhaisse',
    author_email='briandelhaisse@gmail.com',
    maintainer='Brian Delhaisse',
    maintainer_email='briandelhaisse@gmail.com',
    license='(c) Brian Delhaisse',
    url='https://github.com/robotlearn/pyrobolearn',
    platforms=['Linux Ubuntu'],
    # python_requires='==2.7.*',
    # packages=['', 'worlds', 'robots', 'tools', 'pso', 'envs', 'algos', 'optim', 'tasks', 'tools.vr', 'tools.vr.htc', 'tools.vr.oculus',
    #           'tools.bci', 'tools.audio', 'tools.camera', 'tools.game_controllers', 'utils', 'utils.data_structures',
    #           'models', 'robots', 'robots.ros', 'robots.ros.coman', 'robots.ros.walkman', 'filters',
    #           'metrics', 'rl', 'mechanics', 'objectives', 'experiments', 'exploration'],
    packages=find_packages(),  #find_packages(exclude=('tests',))
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
