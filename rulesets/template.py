from typing import Any
import os, colorsys

if __name__ != '__exec__':
    os.chdir('../')
    from complex_analysis import RulesetDataStructure

class Ruleset(RulesetDataStructure):
    # Defines a ruleset for iteration.

    # Attributes
    # self.variables is a dict[str, Any] for extra variables needed during iteration. It is reset using reset_variables
    #     before rendering each pixel.
    # self.max_iterations is the maximum number of iterations being used for the render (this should be read-only).

    # Common Function Arguments
    # i      -> the iterator (i.e. "z" from the Mandelbrot Set formula)
    # pixel  -> the complex number representing the pixel being rendered (i.e. "c" from the Mandelbrot Set formula)
    # inputs -> a list of all the complex number inputs given by the user

    # Configuration for complex user inputs. These are complex numbers that can be modified by the user, and are
    #     passed as the "inputs" argument to functions. They also are shown as colored circles on top of the renders.
    #     There is also a text label that says whatever "name" is on the circle.
    # It should be a list of dicts, containing this data:
    # {'default_value': <the default value for the input (complex)>, 'name': <a name for the input (str)>, 'color': <an RGBA-255 tuple>}
    # There are also six optional keys:
    # {'radius': <radius of the circle in pixels (float)>, 'outline_color': <an RGBA-255 tuple>,
    #     'outline_width': <width of the outline in pixels (int)>, 'text_color': <an RGBA-255 tuple>,
    #     'display_color': <color of the value display in the corner of the screen; an RGBA-255 tuple, or None to hide>,
    #     'display_size': <font size for the value display (float)>}
    # These optional keys default to these values:
    # {'radius': 10, 'outline_color': (0, 0, 0, 255), 'outline_width': 1, 'text_color': (0, 0, 0, 255),
    #     'display_color': (255, 255, 255, 255), 'display_size': 20}
    # This is optional and defaults to an empty list, meaning the set has no user inputs.

    # Example for the Mandelbrot Set:
    inputs_config: list[dict[str, Any]] = [
        {'default_value': 0 + 0j, 'name': 'z0', 'color': (128, 128, 128, 255)}
    ]

    def iteration_rule(self, i: complex, pixel: complex, inputs: list[complex]) -> complex | None:
        # Should return the new value of i, or None. If None, iteration will stop.

        # Example for the Mandelbrot Set:
        if abs(i) > 2:
            return None

        i = i * i + pixel
        return i

    def initial_value(self, pixel: complex, inputs: list[complex]) -> complex:
        # Should return the initial value of i, given the pixel being iterated.
        # This function is optional and defaults to returning 0.

        # Example for the Mandelbrot Set:
        return inputs[0] # this example allows the user to change the initial value of z by using the z0 input

    def color_rule(self, i: complex, iteration_count: int, inputs: list[complex]) -> tuple[int, int, int, int]:
        # Should return an RGBA-255 tuple.
        # This function is optional and defaults to the Mandelbrot Set coloring behavior shown below.

        # Example for the Mandelbrot Set:
        # color black if reached max iteration count
        if iteration_count >= self.max_iterations:
            return 0, 0, 0, 255

        # otherwise rotate hue based on iteration count
        rgb_float: tuple[float, float, float] = colorsys.hsv_to_rgb((iteration_count / 100) % 1, 1, 1)
        return round(rgb_float[0] * 255), round(rgb_float[1] * 255), round(rgb_float[2] * 255), 255

    def reset_variables(self):
        # Add any logic needed for resetting "self.variables" here.
        # This function is optional and defaults to clearing the dict.
        self.variables = {}
