# pyOCR - Machine Learning Core Photograph Renaming Program
Program for automatically renaming drill core photographs using machine learning algorithms to detect drill depths and other keywords handwritten on whiteboards. Includes a UI 
for interactively stepping through photographs and easily making required modifications.

Overview of program steps under the hood: 
* Initial browse user interface for selection of directory containing core photographs
* Detectron2 trained model to detect and isolate the whiteboard in the photos
* CV2 contouring to isolate handwritten characters on whiteboard
* TensorFlow trained neural network to classify the handwritten text characters as alpha-numeric
* Coded algorithms to group characters into words, and to identify keywords "DEPTH", "FROM", and "WET" vs "DRY"
* ScikitLearn logistic regression model to select the correct two words as numbers
* TensorFLow trained neural network to classify the characters in the two numbers as numeric-only
* Main program user interface to step through named photographs interactively

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

## To Run the Program
Double click 'RunProgram.bat' if the environment was setup using setup.bat above. This activates the 'ml' conda environment and executes the python entry point into the program, 'Main.py'. Otherwise, 'python Main.py' in the terminal will run the program. 

Output images will be saved in a subdirectory created within the user selected input directory named "Output_Named_Images".

## Notes on Using the Program
### Opening browse screen
<img src="/input/OpeningBrowseScreen.png" alt="Opening Browse Screen" width="1000"/>

*	Select “Skip machine learning” if you do not want the program to attempt to name photographs for you, you just wish to quickly step through the photos and name them yourself using the program interface. Selecting this option will save you the time of waiting for the machine learning algorithms to compute.
*	Process graphics using ‘GPU’ is recommended if you have an NVIDIA graphics card as computation times will be reduced. However selecting ‘GPU’ may cause your computer to crash if you do not have an NVIDIA graphics card and drivers installed. ‘CPU’ is therefore the default.
### Main program interface
<img src="/input/MainProgramInterface.png" alt="Main Program Interface" width="1000"/>

### Guidance for core photographs for best machine learning algorithm results
*	Best to make sure the white board is totally contained within the photograph
*	Clean "white" whiteboards without black smudges, except for where the text is written
*	Neat writing
*	Write in capitals
*	Use black marker, or if not, the darker the better
*	Do not let characters touch each other at all
*	Leave sufficient space between words such that individual words can be distinguished
*	Conversely do not leave too large of a gap between number characters, especially across the decimal place
*	Do not include 'm' suffix on the actual depth values (i.e. have it in brackets on the line above instead)
*	Must have keywords ‘DEPTH’ and/or 'FROM', and also 'WET' vs 'DRY', as the program searches for these

Note: Examples of good photographs are included in the ‘ExamplePhotos’ directory.

## Program Directory Structure
Files in the root pyOCR directory are part of the main program, and are required to run the program. 
This includes python source code files and machine learning models.

Subdirectories:
* ExamplePhotos: contains example input core photographs.
* input: contains input images and icons used by the program and the README.
* roboflow: contains the json files needed for the whiteboard detection model.
* supplementary code: contains the python source code created for training models, not needed to run the 
program with the models as-is but needed to updated the models.