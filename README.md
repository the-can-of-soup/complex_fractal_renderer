# Complex Fractal Renderer

A Python program for rendering video or still frames of fractals that use complex numbers, such as the Mandelbrot Set or Newton Fractals. 

## Installation
1. Install Python.
2. Install Pillow (`pip install pillow`).
3. Install OpenCV (`pip install opencv-python`).
4. Clone the repo.
5. Run `main.py` at least once. You can close the window after text appears in the terminal.

## How to Use
1. Create a ruleset (fractal), or choose an existing one. This should be a Python file in the `rulesets` folder. See `rulesets/template.py` for information about how to create a ruleset.\
   Some built-in rulesets are the Mandelbrot Set and Newton Fractals. 
2. If you are making a video, create an animation. This should be a JSON file in the `animation_rules` folder. See `animation_rules/template.json` for information about how to create an animation.
3. Run `main.py` and select your render options.
4. Finished renders will be saved in the `renders` folder as an AVI file (video) or PNG file (still frame).
