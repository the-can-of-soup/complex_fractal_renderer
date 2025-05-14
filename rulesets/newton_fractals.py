from typing import Any
import os, math

if __name__ != '__exec__':
    os.chdir('../')
    from complex_analysis import RulesetDataStructure

ROOT_COLORS = [(255, 64, 64, 255), (32, 32, 160, 255), (160, 64, 160, 255), (64, 255, 64, 255)]
# other colors that can be added to add more roots
# (64, 160, 160, 255), (255, 255, 64, 255), (255, 255, 255, 255)
ROOT_DEFAULT_VALUES = [0.8 + 0j, -0.6 - 0.6j, -0.6 + 0.6j, 0.2 - 1.2j, 0.2 + 1.2j, 0.5 - 0.9j, 0.5 + 0.9j]

def evaluate_polynomial(z: complex, inputs: list[complex]) -> complex:
    # evaluates the polynomial with the roots given by "inputs"
    value: complex = 1 + 0j
    for root in inputs:
        value *= z - root
    return value

def evaluate_polynomial_derivative(z: complex, inputs: list[complex]) -> complex:
    # evaluates the derivative of the polynomial with the roots given by "inputs"
    value: complex = 0 + 0j
    for i in range(len(inputs)):
        term: complex = 1 + 0j
        for j in range(len(inputs)):
            if j != i:
                term *= z - inputs[j]
        value += term
    return value

class Ruleset(RulesetDataStructure):
    inputs_config: list[dict[str, Any]] = [{'default_value': ROOT_DEFAULT_VALUES[i], 'name': f'r{i+1}', 'color': ROOT_COLORS[i]} for i in range(len(ROOT_COLORS))]

    def iteration_rule(self, i: complex, pixel: complex, inputs: list[complex]) -> complex | None:
        return i - evaluate_polynomial(i, inputs) / evaluate_polynomial_derivative(i, inputs)

    def initial_value(self, pixel: complex, inputs: list[complex]) -> complex:
        return pixel

    def color_rule(self, i: complex, iteration_count: int, inputs: list[complex]) -> tuple[int, int, int, int]:
        closest: int | None = None
        closest_distance: float = math.inf

        for j in range(len(inputs)):
            root: complex = inputs[j]
            distance: float = abs(i - root)
            if distance < closest_distance:
                closest = j
                closest_distance = distance

        if closest is None:
            closest = 0
        return ROOT_COLORS[closest]

    def reset_variables(self):
        self.variables = {}
