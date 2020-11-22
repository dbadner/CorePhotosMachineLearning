# pyOCR
Program for automatically renaming drill core photographs using machine learning algorithms to detect drill depths and other keywords handwritten on whiteboards. Includes a UI 
for interactively stepping through photographs and easily making required modifications.

## Requirements
* Python 3.7
* Numpy
* Opencv
* h5py
* tensorflow
* tk
* pillow
* cudatoolkit
* cython
* pycocotools (pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI")
* detectron2 (pip install "git+https://github.com/DGMaxime/detectron2-windows.git")
* imutils
* scikit-learn

## Instructions For Setup on Windows
1. Install anaconda: https://www.anaconda.com/products/individual
2. Install git: https://git-scm.com/download/win
3. Install Cuda toolkit: https://developer.nvidia.com/cuda-10.1-download-archive-base
4. Install C++ build tools: https://visualstudio.microsoft.com/visual-cpp-build-tools (run this program and install C++ build tools)
5. Double click setup.bat; this will create a new conda environment named ‘ml’ with python 3.7, will activate that environment, and will install necessary packages


