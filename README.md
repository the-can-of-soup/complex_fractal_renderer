# Complex Fractal Renderer

A Python program for rendering video or still frames of fractals that use complex numbers, such as the Mandelbrot Set or Newton Fractals. 

## Installation
1. Install Python or PyPy.
2. Install Pillow (`pip install pillow` or `pypy -m pip install pillow`).
3. Install OpenCV (`pip install opencv-python`).
4. Clone the repo.
5. Run `main.py` at least once. You can close the window after text appears in the terminal.

## Note for PyPy Users
Because `opencv-python` doesn't work with PyPy, don't try to install it for PyPy. The portion of the program that uses OpenCV will automatically be spawned as a separate process running with CPython. Instead, run `pip install opencv-python` to install it for your CPython installation.

## How to Use
1. Create a ruleset (fractal), or choose an existing one. If creating your own, make a copy of `template.py` in the `rulesets` folder, name it whatever you want, and modify the `Ruleset` class' contents.\
   Some built-in rulesets are the Mandelbrot Set and Newton Fractals.
2. If you are making a video, create an animation. This should be a JSON file in the `animation_rules` folder. See `animation_rules/template.json` for information about how to create an animation.
3. Run `main.py` and select your render options.
4. Finished renders will be saved in the `renders` folder as an AVI file (video) or PNG file (still frame).
