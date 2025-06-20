from setuptools import setup, find_packages

setup(
    name="vps",
    version="0.1.0",
    description="Visual Positioning System combining Hloc and VGGT",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "opencv-python",
        "numpy",
        "pycolmap",
        "h5py",
        "tqdm",
        "matplotlib",
        "scipy",
        "pyyaml",
        "flask",
        "requests",
        "trimesh",
        "pyrender",
        "imageio",
        "pillow"
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'vps-service=vps.service:main',
            'vps-convert=scripts.colmap_to_vps:main',
        ],
    },
) 