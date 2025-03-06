# Assignment 2

This repository contains the source code and images for Assignment 2.

## Structure
  - `images`The folder that has the images, make sure to keep it in the same folder with other scripts and the notebook.

  - `utils.py` Contains the necessary functions for all of the tasks
  - the other scripts are for assignments.

## Assignments 

### Part 1
Run `a1.ipynb`, which applies the gray-scaling algorithms to images and you will also see qualitativ and quantitativ analysis (mse) there.

### Part 2
Run `a2.py`, it provides an interactive tool for applying 3 different quantization techniques.
It is recommended to resize images for faster calculation, as **median cut ** and ** K-means** quantization requires more calculation rather than uniform quantization.

### Part 3
Run `a3.py`, it provides an interactive tool for changing the hue, saturation, brightnes and lightness of the image in real time.

### Part 4
Run `a3.py`, it provides an interactive tool for picking a point and colours all the pixels which have a smaller delta E distance than threshold with  the **label colour** 

## How to run

~~~
git clone https://github.com/ADA-GWU/a2-digital-images-and-color-spaces-aghayevagha.git
cd a2-digital-images-and-color-spaces-aghayevagha
~~~

Install the necessary libraries
~~~
pip install opencv-python numpy matplotlib scikit-image scikit-learn
~~~

  - For the assignment one, run the jupyter notebook `a1.ipynb`.
  - run `a2.py`, `a3.py`, `a4.py` for corresponding parts of the assignments, example:
~~~
python a2.py
~~~

## Credits
The image number five is taken from the internet ([source](https://www.istockphoto.com/stock-photos/nature-and-landscapes))
