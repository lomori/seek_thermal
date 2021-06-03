from setuptools import setup

package_name = 'seek_thermal'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Luiz Omori',
    maintainer_email='luiz.omori@gmail.com',
    description='ROS node for Seek Thermal cameras',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seek_coral = seek_thermal.seek_coral:main',
        ],
    },
)
