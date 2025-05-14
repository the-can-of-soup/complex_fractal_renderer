from typing import Any
import os

if __name__ != '__exec__':
    os.chdir('../')
    from complex_analysis import RulesetDataStructure

class Ruleset(RulesetDataStructure):
    inputs_config: list[dict[str, Any]] = [
        {'default_value': 0 + 0j, 'name': 'z0', 'color': (128, 128, 128, 255)}
    ]

    def iteration_rule(self, i: complex, pixel: complex, inputs: list[complex]) -> complex | None:
        if abs(i) > 2:
            return None

        i = i * i + pixel
        return i

    def initial_value(self, pixel: complex, inputs: list[complex]) -> complex:
        return inputs[0]  # allows the user to change the initial value of z by using the z_0 input
