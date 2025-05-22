from PIL import Image, ImageDraw, ImageFont
from typing import Any, Type
import traceback
import colorsys
import platform
import json
import math
import time
import os

BLOCK_CHARACTERS: list[str] = [' ', '░', '▒', '▓', '█']
if platform.python_implementation() == 'PyPy': # normal block characters break in PyPy for some reason
    BLOCK_CHARACTERS: list[str] = [' ', '.', '-', '=', '#']

# noinspection SpellCheckingInspection
FONT_PATH: str = 'C:/Windows/Fonts/courbd.ttf'
VALID_EASING_TYPES: list[str] = ['instant', 'linear', 'in', 'out', 'in_out']
VALID_SUB_ACTION_PROPERTY_NAMES: list[str] = ['position', 'r_offset', 'i_offset', 'zoom', 'zoom_log', 'inputs']
FOLDER_NAMES: list[str] = ['animation_rules', 'frames', 'renders', 'rulesets']

def clear_screen() -> None:
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')

def get_complex_from_pixel(x: float, y: float, width: int, height: int, real_offset: float = 0, imaginary_offset: float = 0, zoom: float = 100) -> complex:
    # find the complex number representing the center of pixel (x,y) in an image of size (width,height)
    # zoom represents number of pixels per 1 unit
    real: float = (x - ((width - 1) / 2)) / zoom + real_offset
    imaginary: float = (-y + ((height - 1) / 2)) / zoom + imaginary_offset
    return complex(real, imaginary)

def get_pixel_from_complex(z: complex, width: int, height: int, real_offset: float = 0, imaginary_offset: float = 0, zoom: float = 100) -> tuple[float, float]:
    # find the coordinates of the pixel representing complex number z
    # zoom represents number of pixels per 1 unit
    x: float = zoom * (z.real - real_offset) + ((width - 1) / 2)
    y: float = zoom * (imaginary_offset - z.imag) + ((height - 1) / 2)
    return x, y

def show_progress_bar(text: str, progress: float, finished: bool = False, start_time: float | None = None):
    # settings
    num_block_characters: int = len(BLOCK_CHARACTERS)
    progress_bar_length: int = 50

    # generate main progress bar
    progress_bar: str = ''
    for i in range(progress_bar_length):
        block_progress: float = progress * progress_bar_length - i
        if block_progress > 1:
            block_progress = 1
        elif block_progress < 0:
            block_progress = 0

        block_character: str = BLOCK_CHARACTERS[math.floor(block_progress * (num_block_characters - 1))]
        progress_bar += block_character

    # generate ETA
    eta: str = ''
    if start_time is not None and 0 < progress < 1:
        now: float = time.time()
        delta_time: float = now - start_time
        if delta_time > 10: # only show ETA after 10 seconds have passed to give a more accurate prediction
            estimated_remaining: float = (delta_time / progress) * (1 - progress)
            eta = f' (ETA {str(math.floor(estimated_remaining / 3600)).zfill(2)}:{str(math.floor(estimated_remaining / 60) % 60).zfill(2)}:{str(math.floor(estimated_remaining) % 60).zfill(2)})'

    # print to console
    end: str = '\r'
    if finished:
        end = '\n'
    print(f'{text} [{progress_bar}] {math.floor(progress * 100000) / 1000:.3f}%{eta}                   ', end=end)

def interpolate(a: float, b: float, t: float, easing_type: str = 'linear') -> float:
    eased_t: float = 0

    if easing_type == 'linear':
        eased_t = t
    elif easing_type == 'in':
        eased_t = t * t
    elif easing_type == 'out':
        eased_t = t * (2 - t)
    elif easing_type == 'in_out':
        if t < 0.5:
            eased_t = 2 * t * t
        else:
            eased_t = (4 - 2 * t) * t - 1

    return eased_t * (b - a) + a

def interpolate_list(a: float | list[float] | list[list[float]], b: float | list[float] | list[list[float]], t: float, easing_type: str = 'linear') -> float | list[float] | list[list[float]]:
    if is_float(a):
        return interpolate(a, b, t, easing_type)
    if isinstance(a, list):
        if len(a) > 0:
            if isinstance(a[0], list):
                # handle nested lists
                # noinspection PyUnresolvedReferences,PyTypeChecker
                return [[interpolate(a[i][j], b[i][j], t, easing_type) for j in range(len(a[i]))] for i in range(len(a))]

        # handle lists
        return [interpolate(a[i], b[i], t, easing_type) for i in range(len(a))]

    return a

def is_float(x: float) -> bool:
    return isinstance(x, float) or isinstance(x, int)

def resized_font_to_fit_in_circle(draw: ImageDraw.ImageDraw, text: str, radius: float) -> ImageFont.FreeTypeFont:
    size: int = 1
    font: ImageFont = ImageFont.truetype(FONT_PATH, size)
    width: int = 0
    height: int = 0
    while math.sqrt((width / 2) ** 2 + (height / 2) ** 2) < radius - 2 or size == 1:
        font: ImageFont = ImageFont.truetype(FONT_PATH, size)
        bounding_box: tuple[int, int, int, int] = draw.textbbox((0, 0), text, font)
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]
        size += 1
    return font

def complex_number_display(z: complex) -> str:
    # base strings
    real_part: str = f'{str(int(math.floor(abs(z.real)))).zfill(2)}.{str(int(math.floor((abs(z.real) * 1000000) % 1000000))).zfill(6)}'
    imag_part: str = f'{str(int(math.floor(abs(z.imag)))).zfill(2)}.{str(int(math.floor((abs(z.imag) * 1000000) % 1000000))).zfill(6)}'

    # real part negative sign
    if z.real < 0:
        real_part = '-' + real_part
    else:
        real_part = ' ' + real_part

    # imaginary part negative sign and combine the parts
    if z.imag < 0:
        return f'{real_part} - {imag_part}i'
    return f'{real_part} + {imag_part}i'

class RulesetDataStructure:
    # This class should be used as the parent class for Rulesets (see "rulesets/template.py").

    def __init__(self):
        self.variables: dict[str, Any] = {}
        self.max_iterations: int = 0

    def iterate(self, pixel: complex, inputs: list[complex], max_iterations: int = 1024) -> tuple[int, int, int, int]:
        # reset variables
        self.reset_variables()

        # save max_iterations as an attribute of self
        self.max_iterations = max_iterations

        # apply iteration rule repeatedly
        i: complex = self.initial_value(pixel, inputs)
        iteration: int = 0
        while iteration < max_iterations:
            new_i: complex = self.iteration_rule(i, pixel, inputs)
            if new_i is None:
                break
            i = new_i
            iteration += 1

        # apply color rule to get color
        return self.color_rule(i, iteration, inputs)

    def render(self, width: int, height: int, real_offset: float = 0, imaginary_offset: float = 0, zoom: float = 100, inputs: list[complex] | None = None, max_iterations: int = 1024, show_inputs: bool = False, show_displays: bool = False) -> Image:
        show_progress_bar('Frame preparing... ', 0)

        # by default set all inputs to their configured default values
        if inputs is None:
            inputs: list[complex] = self.default_inputs

        # initialize image
        im: Image = Image.new('RGBA', (width, height), (255, 255, 255, 255))

        # iterate each pixel
        start_time: float = time.time()
        for y in range(height):
            show_progress_bar('Frame progress:    ', y / height, start_time=start_time)
            for x in range(width):
                pixel: complex = get_complex_from_pixel(x, y, width, height, real_offset, imaginary_offset, zoom)
                color: tuple[int, int, int, int] = self.iterate(pixel, inputs, max_iterations)
                im.putpixel((x, y), color)

        show_progress_bar('Frame finalizing...', 1)

        # create ImageDraw object
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(im)

        # add circles to show inputs if enabled
        if show_inputs:
            for i in range(len(inputs)):
                # get config and position
                config: dict[str, Any] = self.__class__.inputs_config[i]
                value: complex = inputs[i]
                xy: tuple[float, float] = get_pixel_from_complex(value, width, height, real_offset, imaginary_offset, zoom)
                radius: float = 10
                if 'radius' in config:
                    radius = config['radius']
                outline_color: tuple[int, int, int, int] = (0, 0, 0, 255)
                if 'outline_color' in config:
                    outline_color = config['outline_color']
                outline_width: int = 1
                if 'outline_width' in config:
                    outline_width = config['outline_width']
                text_color: tuple[int, int, int, int] = (0, 0, 0, 255)
                if 'text_color' in config:
                    text_color = config['text_color']

                # draw circle and text
                draw.circle(xy, radius, fill=config['color'], outline=outline_color, width=outline_width)
                draw.text(xy, config['name'], text_color, font=resized_font_to_fit_in_circle(draw, config['name'], radius), anchor='mm', align='center')

        # add displays to show inputs' values if enabled
        if show_displays:
            # loop through inputs first to find the longest name
            longest_name_length: int = 0
            for i in range(len(inputs)):
                config: dict[str, Any] = self.__class__.inputs_config[i]
                if 'display_color' in config:
                    if config['display_color'] is None:
                        continue

                if len(config['name']) > longest_name_length:
                    longest_name_length = len(config['name'])

            # start at coordinates (5, 5) and draw each line of text individually
            xy: tuple[float, float] = (5, 5)
            for i in range(len(inputs)):
                # get config
                config: dict[str, Any] = self.__class__.inputs_config[i]
                display_color: tuple[int, int, int, int] | None = (255, 255, 255, 255)
                if 'display_color' in config:
                    display_color = config['display_color']
                display_size: float = 20
                if 'display_size' in config:
                    display_size = config['display_size']

                # don't draw if input's display is disabled
                if display_color is None:
                    continue

                # generate line of text
                value: complex = inputs[i]
                text: str = f'{config["name"].ljust(longest_name_length)} = {complex_number_display(value)}'

                # draw
                font: ImageFont.FreeTypeFont = ImageFont.truetype(FONT_PATH, display_size)
                draw.text(xy, text, display_color, font, anchor='lt')
                ascent, descent = font.getmetrics()
                xy = (xy[0], xy[1] + ascent + descent)

        show_progress_bar('Frame finished!    ', 1, True)

        # return image
        return im

    @property
    def default_inputs(self) -> list[complex]:
        inputs: list[complex] = []
        for i in self.__class__.inputs_config:
            inputs.append(i['default_value'])
        return inputs

    # \/ These methods below can/should be modified by child classes. \/
    # ------------------------------------------------------------------

    inputs_config: list[dict[str, Any]] = []

    def iteration_rule(self, i: complex, pixel: complex, inputs: list[complex]) -> complex | None:
        return None

    def initial_value(self, pixel: complex, inputs: list[complex]) -> complex:
        return 0 + 0j

    def color_rule(self, i: complex, iteration_count: int, inputs: list[complex]) -> tuple[int, int, int, int]:
        # This is the default behavior and is overridden by some sets.

        # color black if reached max iteration count
        if iteration_count >= self.max_iterations:
            return 0, 0, 0, 255

        # otherwise rotate hue based on iteration count
        rgb_float: tuple[float, float, float] = colorsys.hsv_to_rgb((iteration_count / 100) % 1, 1, 1)
        return round(rgb_float[0] * 255), round(rgb_float[1] * 255), round(rgb_float[2] * 255), 255

    def reset_variables(self):
        self.variables = {}

class Animation:
    # An animation defined by rules set by a JSON file in "animation_rules/" (see "animation_rules/template.json").

    def __init__(self, rules: dict[str, Any]):
        # validate rules
        self.rules: dict[str, Any] = rules
        highest_referenced_input_id, inputs_list_length = self.validate()

        # get ruleset by running ruleset Python file and retrieving "Ruleset" class
        path_to_file: str = os.path.join('rulesets', rules['ruleset'])
        exec_namespace: dict[str, Any] = {'__name__': '__exec__', 'RulesetDataStructure': RulesetDataStructure}
        with open(path_to_file, 'r') as f:
            code: str = f.read()
        exec(code, exec_namespace)
        assert 'Ruleset' in exec_namespace, f"Ruleset \"{rules['ruleset']}\" doesn't have a class named \"Ruleset\"!"
        self.ruleset_type: Type[RulesetDataStructure] = exec_namespace['Ruleset']
        self.ruleset: RulesetDataStructure = self.ruleset_type()
        num_inputs: int = len(self.ruleset_type.inputs_config)
        assert highest_referenced_input_id < num_inputs, f'"input_{highest_referenced_input_id}" referenced, but highest ID is "input_{num_inputs}"!'
        assert inputs_list_length == num_inputs or inputs_list_length is None, f'Value must be a length {num_inputs} list (because the ruleset has {num_inputs} inputs) for property "inputs", but a length {inputs_list_length} list was found!'

        # set attributes
        self.frame_rate: int = rules['frame_rate']
        self.resolution: tuple[int, int] = rules['resolution']
        self.max_iterations: int = rules['max_iterations']
        self.show_inputs: bool = rules['show_inputs']
        self.show_displays: bool = rules['show_displays']
        self.position: tuple[float, float, float] = (0, 0, 100) # (real_offset, imaginary_offset, zoom)
        self.inputs: list[complex] = self.ruleset.default_inputs # inputs for rendering frames
        self.current_action_id: int = 0 # the ID of the action currently being rendered (see "animation_rules/template.json")
        self.rendered_frames: int = 0
        self.render_start_time: float | None = None

        # calculate length of whole animation in frames
        self.length: int = 1 # length has 1 added because one more frame is added in the render() function
        for action in self.rules['actions']:
            self.length += round(action['length'] * self.frame_rate)

    def show_progress_bar(self, extra: str | None = None):
        clear_screen()
        print('COMPLEX FRACTAL RENDERER')
        print('--------------------')
        print('')
        print('Rendering animation...')
        if extra is not None:
            print(extra)

        # finished set to true so that print doesn't end with \r
        show_progress_bar('Animation progress:', self.rendered_frames / self.length, True, self.render_start_time)

    def render_frame(self) -> Image:
        return self.ruleset.render(self.resolution[0], self.resolution[1],
                                   self.position[0], self.position[1], self.position[2],
                                   self.inputs, self.max_iterations, self.show_inputs, self.show_displays)

    def save_frame(self, frame: Image):
        frame.save(f'frames/frame_{str(self.rendered_frames).zfill(6)}.png')
        self.rendered_frames += 1

    def get_property(self, property_name: str) -> float | list[float] | list[list[float]]:
        if property_name == 'position':
            return list(self.position)
        elif property_name == 'r_offset':
            return self.position[0]
        elif property_name == 'i_offset':
            return self.position[1]
        elif property_name == 'zoom':
            return self.position[2]
        elif property_name == 'zoom_log':
            return math.log10(self.position[2])
        elif property_name == 'inputs':
            return [[z.real, z.imag] for z in self.inputs]
        elif property_name.startswith('input_'):
            z: complex = self.inputs[int(property_name[6:])]
            return [z.real, z.imag]

    def set_property(self, property_name: str, value: float | list[float] | list[list[float]]):
        if property_name == 'position': # value is list[float]
            self.position = tuple(value)
        elif property_name == 'r_offset': # value is float
            self.position = value, self.position[1], self.position[2]
        elif property_name == 'i_offset': # value is float
            self.position = self.position[0], value, self.position[2]
        elif property_name == 'zoom': # value is float
            self.position = self.position[0], self.position[1], value
        elif property_name == 'zoom_log': # value is float
            self.position = self.position[0], self.position[1], 10 ** value
        elif property_name == 'inputs': # value is list[list[float]] (sub-lists represent complex numbers)
            self.inputs = []
            for z in value:
                self.inputs.append(complex(z[0], z[1]))
        elif property_name.startswith('input_'): # value is list[float] (represents a complex number)
            self.inputs[int(property_name[6:])] = complex(value[0], value[1])

    def render_action(self, action_id: int):
        # collect data from action
        action: dict[str, Any] = self.rules['actions'][action_id]
        easing_type: str = action['easing']
        action_length: int = round(action['length'] * self.frame_rate) # length of action in frames
        sub_actions: list[dict[str, Any]] = action['data']

        # render action
        if easing_type == 'instant':
            # set properties
            for sub_action in sub_actions:
                self.set_property(sub_action['name'], sub_action['value'])

            # wait for "action_length" frames by writing the same frame over and over
            if action_length > 0:
                self.show_progress_bar(f'{self.rendered_frames} / {self.length} frames rendered; rendering action #{action_id} with easing "{easing_type}"')
                im: Image = self.render_frame()

                for i in range(action_length):
                    self.show_progress_bar(f'{self.rendered_frames} / {self.length} frames rendered; rendering action #{action_id} with easing "{easing_type}"')

                    self.save_frame(im)
        else:
            # save the start and end values of the sub-actions
            start_values: list[float | list[float] | list[list[float]]] = [self.get_property(sub_action['name']) for sub_action in sub_actions]
            end_values: list[float | list[float] | list[list[float]]] = [sub_action['value'] for sub_action in sub_actions]

            # loop 1 extra time to calculate values where t = 1, but don't actually render the frame on the last loop
            for i in range(action_length + 1):
                # show progress bar
                if i < action_length:
                    self.show_progress_bar(f'{self.rendered_frames} / {self.length} frames rendered; rendering action #{action_id} with easing "{easing_type}"')

                # calculate t based on how far the loop is through the action
                t: float = i / action_length

                # interpolate values
                for j in range(len(sub_actions)):
                    sub_action: dict[str, Any] = sub_actions[j]
                    start: float | list[float] | list[list[float]] = start_values[j]
                    end: float | list[float] | list[list[float]] = end_values[j]
                    self.set_property(sub_action['name'], interpolate_list(start, end, t, easing_type))

                # render frame
                if i < action_length:
                    im: Image = self.render_frame()
                    self.save_frame(im)

    def render(self):
        # record start time for progress bar ETA
        self.render_start_time = time.time()

        # get list of actions
        actions: list[dict[str, Any]] = self.rules['actions']

        # render actions
        for action_id in range(len(actions)):
            self.render_action(action_id)

        self.show_progress_bar(f'{self.rendered_frames} / {self.length} frames rendered; rendering final frame')

        # add one last frame because actions only render up until the end, not actually the end
        im: Image = self.render_frame()
        self.save_frame(im)

        self.show_progress_bar(f'{self.rendered_frames} / {self.length} frames rendered; finished')

    def validate(self) -> tuple[int, int | None]:
        # We can assume...
        # The data structure is valid parsed JSON.

        assert isinstance(self.rules, dict), 'Animation data must be a JSON object!'

        # The data structure is a dict.

        required_keys: dict[str, type | None] = {
            'ruleset': str,
            'frame_rate': int,
            'resolution': None, # "resolution" has a special type handler
            'max_iterations': int,
            'show_inputs': bool,
            'show_displays': bool,
            'actions': list
        }

        for key in required_keys:
            assert key in self.rules, f'Animation data must have key "{key}"!'
            required_type: type | None = required_keys[key]
            if required_type is None:
                continue
            assert isinstance(self.rules[key], required_type), f'Key "{key}" must be of type {str(required_type)}!'

        # All necessary keys exist and all except "resolution" are the correct type.

        assert isinstance(self.rules['resolution'], list), 'Key "resolution" must be a list!'
        assert len(self.rules['resolution']) == 2, 'Key "resolution" must be a length 2 list!'
        assert isinstance(self.rules['resolution'][0], int) and isinstance(self.rules['resolution'][1], int), 'Key "resolution" must be a list of 2 ints!'

        # All keys are the correct type.

        assert '..' not in self.rules['ruleset'], 'Key "ruleset" may not contain ".."!'
        assert ':' not in self.rules['ruleset'], 'Key "ruleset" may not contain ":"!'
        assert not (self.rules['ruleset'].startswith('/') or self.rules['ruleset'].startswith('\\')), 'Key "ruleset" may not start with a slash!'

        # Key "ruleset" is fully valid.

        assert 0 < self.rules['frame_rate'] <= 60, 'Key "frame_rate" must be in the range (0,60]!'
        assert 0 < self.rules['resolution'][0] <= 3840 and 0 < self.rules['resolution'][1] <= 2160, 'Resolution must be between 0x0 and 3840x2160!'
        assert 0 <= self.rules['max_iterations'] <= 100000, 'Max iterations must be in the range [0,100000]!'

        # All keys except "actions" are fully valid.

        highest_referenced_input_id: int = -1
        inputs_list_length: int | None = None
        for i in range(len(self.rules['actions'])):
            action: dict = self.rules['actions'][i]
            assert isinstance(action, dict), f'Action #{i} must be a dict!'

            # The action is a dict.

            assert 'easing' in action, f'Action #{i} must have the key "easing"!'
            assert 'length' in action, f'Action #{i} must have the key "length"!'
            assert 'data' in action, f'Action #{i} must have the key "data"!'

            # The action has all necessary keys.

            assert isinstance(action['easing'], str), f'Key "easing" of action #{i} must be a str!'
            assert is_float(action['length']), f'Key "length" of action #{i} must be a float!'
            assert isinstance(action['data'], list), f'Key "data" of action #{i} must be a list!'

            # All keys are the correct type.

            assert action['easing'] in VALID_EASING_TYPES, f'Invalid easing type "{action["easing"]}" in action #{i}!'
            assert 0 <= action['length'] <= 1000, f'Key "length" of action #{i} must be in the range [0,1000]!'

            # All keys except "data" are fully valid.

            for j in range(len(action['data'])):
                sub_action: dict = action['data'][j]
                assert isinstance(sub_action, dict), f'Sub-action #{j} in action #{i} must be a dict!'

                # The sub-action is a dict.

                assert 'name' in sub_action, f'Sub-action #{j} in action #{i} must have the key "name"!'
                assert 'value' in sub_action, f'Sub-action #{j} in action #{i} must have the key "value"!'
                property_name: str = sub_action['name']

                # The sub-action has all necessary keys.

                assert isinstance(property_name, str), f'Key "name" in sub-action #{j} in action #{i} must be a str!'
                if property_name.startswith('input_'):
                    is_valid_int: bool = True
                    integer: int = 0
                    try:
                        integer = int(property_name[6:])
                    except (ValueError, IndexError):
                        is_valid_int = False
                    assert is_valid_int and integer >= 0, f'Invalid property name "{property_name}" in sub-action #{j} in action #{i}!'
                    if integer > highest_referenced_input_id:
                        highest_referenced_input_id = integer
                else:
                    assert property_name in VALID_SUB_ACTION_PROPERTY_NAMES, f'Invalid property name "{property_name}" in sub-action #{j} in action #{i}!'

                # The sub-action has a valid property name.

                if property_name == 'position':
                    assert isinstance(sub_action['value'], list), f'Value must be a list for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    assert len(sub_action['value']) == 3, f'Value must be a length 3 list for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    assert is_float(sub_action['value'][0]) and is_float(sub_action['value'][1]) and is_float(sub_action['value'][2]), f'Value must be a list of 3 floats for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    assert sub_action['value'][2] > 0, f'Third item of list "value" must be greater than 0 for the property "{property_name}" in sub-action #{j} in action #{i}!'
                elif property_name in ('r_offset', 'i_offset', 'zoom', 'zoom_log'):
                    assert is_float(sub_action['value']), f'Value must be a float for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    if property_name == 'zoom':
                        assert sub_action['value'] > 0, f'Value must be greater than 0 for the property "{property_name}" in sub-action #{j} in action #{i}!'
                elif property_name == 'inputs':
                    assert isinstance(sub_action['value'], list), f'Value must be a list for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    if inputs_list_length is None:
                        inputs_list_length = len(sub_action['value'])
                    elif len(sub_action['value']) != inputs_list_length:
                        raise AssertionError(f'Inconsistent length of list for property "inputs"! Last length: {inputs_list_length}; length for sub-action #{j} in action #{i}: {len(sub_action["value"])}')
                    for input_value in sub_action['value']:
                        assert isinstance(input_value, list), f'Value must be a list of lists for the property "{property_name}" in sub-action #{j} in action #{i}!'
                        assert len(input_value) == 2, f'Value must be a list of length 2 lists for the property "{property_name}" in sub-action #{j} in action #{i}!'
                        assert is_float(input_value[0]) and is_float(input_value[1]), f'Value must be a list of lists of 2 floats for the property "{property_name}" in sub-action #{j} in action #{i}!'
                else: # property name is of the format "input_x" where x is a non-negative integer
                    assert isinstance(sub_action['value'], list), f'Value must be a list for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    assert len(sub_action['value']) == 2, f'Value must be a length 2 list for the property "{property_name}" in sub-action #{j} in action #{i}!'
                    assert is_float(sub_action['value'][0]) and is_float(sub_action['value'][1]), f'Value must be a list of 2 floats for the property "{property_name}" in sub-action #{j} in action #{i}!'

                # The sub-action has a valid value for its property and all keys of the sub-action are valid.
            # All keys of the action are fully valid.
        # All keys are fully valid.

        return highest_referenced_input_id, inputs_list_length

def main():
    # Create folders if they don't exist
    for folder in FOLDER_NAMES:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    while True:
        clear_screen()
        print('COMPLEX FRACTAL RENDERER')
        print('--------------------')
        print('')
        print('Enter "a" to render an animation, or "s" to render a static image.')
        mode = input(' > ').strip().lower()
        print('')
        if mode == 'a':
            # ANIMATION MODE

            clear_screen()
            print('COMPLEX FRACTAL RENDERER')
            print('--------------------')
            print('')
            print('For information about creating an animation, look at')
            print('"animation_rules/template.json" and "rulesets/template.py".')
            print('')
            print('ANIMATIONS')

            # get list of animation files
            animation_filenames: list[str] = os.listdir('animation_rules')
            animation_filenames = [i for i in animation_filenames if os.path.isfile(os.path.join('animation_rules', i))]
            animation_filenames.sort()

            # print list
            for i in range(len(animation_filenames)):
                filename: str = animation_filenames[i]
                print(f'{f"[{str(i)}]".ljust(5)} {filename}')
            print('')

            # ask for an ID and verify it
            print('Enter the ID of an animation above to render it.')
            animation_id: str = input(' > ').strip()
            print('')
            try:
                animation_id: int = int(animation_id)
                assert 0 <= animation_id < len(animation_filenames)
            except (ValueError, AssertionError):
                print('Invalid input!')
                print('')
                input('Press ENTER to continue.')
                continue
            animation_filename: str = animation_filenames[animation_id]

            # load animation rules and create object
            with open(os.path.join('animation_rules', animation_filename), 'r') as f:
                rules_str: str = f.read()
            try:
                rules: dict[str, Any] = json.loads(rules_str)
            except json.decoder.JSONDecodeError as e:
                print(f'JSON SYNTAX ERROR: {e}')
                print(f'Check "{os.path.join("animation_rules", animation_filename)}" to try to fix the error.')
                print('')
                input('Press ENTER to continue.')
                continue
            try:
                animation: Animation = Animation(rules)
            except AssertionError as e:
                print(f'FORMAT ERROR: {e}')
                print(f'Check "{os.path.join("animation_rules", animation_filename)}" to try to fix the error.')
                print('')
                input('Press ENTER to continue.')
                continue

            print('Ready to start rendering! This will clear all old images from the "frames" folder.')
            input('Press ENTER to start.')
            print('')

            # noinspection PyBroadException
            try:
                # clear old frames
                old_frame_filenames: list[str] = os.listdir('frames')
                for filename in old_frame_filenames:
                    os.remove(os.path.join('frames', filename))

                # render frames
                animation.render()

                clear_screen()
                print('COMPLEX FRACTAL RENDERER')
                print('--------------------')
                print('')
                print('Finalizing...')

                # dump data to json file
                output_filename: str = os.path.join('renders', animation_filename.strip('.json') + '.avi')
                with open('video_finalizer_info.json', 'w') as f:
                    f.write(json.dumps({
                        'output_filename': output_filename,
                        'frame_rate': rules['frame_rate'],
                        'resolution': rules['resolution']
                    }))

                # finalize using default Python, not PyPy because cv2 doesn't work with PyPy
                os.system('py finalize_video.py')
                now: float = time.time()
                delta_time: float = now - animation.render_start_time
                print('')
                print(f'Finished and saved to {output_filename}!')
                print(f'Time spent: {str(math.floor(delta_time / 3600)).zfill(2)}:{str(math.floor(delta_time / 60) % 60).zfill(2)}:{str(math.floor(delta_time) % 60).zfill(2)}')
                print('')
                input('Press ENTER to finish.')
                continue
            except:
                print('AN ERROR OCCURRED')
                print('========================================')
                print('')
                traceback.print_exc()
                print('')
                print('========================================')
                print(f'Check "{os.path.join("rulesets", rules["ruleset"])}" to try to fix the error.')
                print('')
                input('Press ENTER to continue.')
                continue

        elif mode == 's':
            # STATIC IMAGE MODE

            clear_screen()
            print('COMPLEX FRACTAL RENDERER')
            print('--------------------')
            print('')
            print('For information about creating a ruleset, look at')
            print('"rulesets/template.py".')
            print('')
            print('RULESETS')

            # get list of animation files
            ruleset_filenames: list[str] = os.listdir('rulesets')
            ruleset_filenames = [i for i in ruleset_filenames if os.path.isfile(os.path.join('rulesets', i))]
            ruleset_filenames.sort()

            # print list
            for i in range(len(ruleset_filenames)):
                filename: str = ruleset_filenames[i]
                print(f'{f"[{str(i)}]".ljust(5)} {filename}')
            print('')

            # ask for an ID and verify it
            print('Enter the ID of a ruleset above to render it.')
            ruleset_id: str = input(' > ').strip()
            print('')
            try:
                ruleset_id: int = int(ruleset_id)
                assert 0 <= ruleset_id < len(ruleset_filenames)
            except (ValueError, AssertionError):
                print('Invalid input!')
                print('')
                input('Press ENTER to continue.')
                continue
            ruleset_filename: str = ruleset_filenames[ruleset_id]

            # define rules for the "animation"
            rules: dict = {
                'ruleset': ruleset_filename,
                'frame_rate': 1,
                'show_inputs': False,
                'show_displays': False,
                'actions': []
            }

            # position
            print('Enter the position and zoom level for the camera in the format "Re,Im,Zoom".')
            print('Zoom should be in pixels per unit.')
            position: list[str] = input(' > ').replace(' ', '').replace(',', 'x').lower().split('x')
            print('')
            try:
                assert len(position) == 3
                position: list[float] = [float(i) for i in position]
                assert position[2] > 0
            except (ValueError, AssertionError):
                print('Invalid input!')
                print('The position should be three numbers, and the third (the zoom) must be more than 0.')
                print('')
                input('Press ENTER to continue.')
                continue
            # noinspection PyTypeChecker
            position: tuple[float, float, float] = tuple(position)

            # max_iterations
            print('Enter the maximum number of iterations.')
            max_iterations: str = input(' > ').strip()
            print('')
            try:
                max_iterations: int = int(max_iterations)
                assert 0 <= max_iterations <= 100000
            except (ValueError, AssertionError):
                print('Invalid input!')
                print('The max iterations should be a whole number in the range [0,100000].')
                print('')
                input('Press ENTER to continue.')
                continue
            rules['max_iterations'] = max_iterations

            # resolution
            print('Enter the resolution for the image in the format "WxH".')
            resolution: list[str] = input(' > ').replace(' ', '').replace(',', 'x').lower().split('x')
            print('')
            try:
                assert len(resolution) == 2
                resolution: list[int] = [int(i) for i in resolution]
                assert 0 < resolution[0] <= 3840 and 0 < resolution[1] <= 2160
            except (ValueError, AssertionError):
                print('Invalid input!')
                print('The resolution should be two whole numbers between 0x0 and 3840x2160.')
                print('')
                input('Press ENTER to continue.')
                continue
            rules['resolution'] = resolution

            # define "animation" object
            animation: Animation = Animation(rules)
            animation.position = position

            if len(animation.ruleset_type.inputs_config) > 0:
                # show_inputs
                print('Would you like the inputs to be visible in the image as colored circles?')
                rules['show_inputs'] = input(' > ').strip().lower() in ('y', 'yes')
                animation.rules['show_inputs'] = rules['show_inputs']
                animation.show_inputs = rules['show_inputs']
                print('')

                # show_inputs
                print('Would you like the values of the inputs to be displayed in the corner?')
                rules['show_displays'] = input(' > ').strip().lower() in ('y', 'yes')
                animation.rules['show_displays'] = rules['show_displays']
                animation.show_displays = rules['show_displays']
                print('')

                # inputs
                print('INPUTS')
                print('Enter inputs as complex numbers in the format Re,Im.')
                print('Invalid inputs will use the default value.')
                for i in range(len(animation.ruleset_type.inputs_config)):
                    input_config: dict[str, Any] = animation.ruleset_type.inputs_config[i]
                    new_value: list[str] = input(f' - {input_config["name"]} > ').replace(' ','').split(',')
                    try:
                        assert len(new_value) == 2
                        new_value: complex = complex(float(new_value[0]), float(new_value[1]))
                    except (AssertionError, ValueError):
                        new_value: complex = input_config['default_value']
                    animation.inputs[i] = new_value
                print('')

            # ask for output file path
            print('Enter the filename for the image to be saved (in "renders/"):')
            output_filename: str = os.path.join('renders', input(' > ').strip())
            print('')

            # render and save frame
            # noinspection PyBroadException
            try:
                animation.render_frame().save(output_filename)
                print('')
                input('Press ENTER to finish.')
                continue
            except:
                print('AN ERROR OCCURRED')
                print('========================================')
                print('')
                traceback.print_exc()
                print('')
                print('========================================')
                print(f'Check "{os.path.join("rulesets", ruleset_filename)}" to try to fix the error.')
                print('')
                input('Press ENTER to continue.')
                continue

        else:
            print('Invalid input!')
            print('')
            input('Press ENTER to continue.')
            continue

if __name__ == '__main__':
    main()
