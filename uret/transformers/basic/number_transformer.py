from uret.transformers import Transformer

import numpy as np
import random


class NumberTransformer(Transformer):
    name = "NumberTransformer"

    def __init__(self, subtransformer_args, input_constraints={}, input_processor=None, number_type="float"):
        """
        Initialize a `NumberTransformer` object.
        :param subtransformer_list: The functions to be used to transform the input of a certain data type.
        :param input_constraints: The constraints to enforce on the input after transformation. The possible enforcement values
        are:
            - "bounds": Should be in the form (lower, upper, method). Mix and max indicate the end points of possible values for
            the input. Method is an optional parameter, which can control the bound clipping behavior. The possible values are:
                0: Always clip the values based on the mix/max bounds. (Default)
                1: Ensure that the value never drops below the base feature.
                2: Ensure that the value never goes above the base feature.
                3: Ensure that the value never drops below the base feature and the value is not clipped up to the lower bound.
                   In other words, never clip up from the base value
                4: Ensure that the value never goes above the base feature and the value is not clipped down to the upper bound.
                   In other words, never clip down from the base value
                5: A combination of 3 and 4. In other words, never clip in from the base value
            - "eps": Should be in the form {"value", "type"}. eps_type is an optional parameter, which controls how the
                     distance value is enforced. The possible type values are
                "abs": The distance value represents the absolute maximum value of the transformation record. (Default)
                "rel": The distance value represents a percentage of the original value should be used to enforce
                       constraints.
        :param input_processor: A function with two modes to control transforming input values where parts of the input should
        remain unchanged. If a single input is given, the function will return (the modifiable sections, unmodifiable sections).
        If two inputs are given, it is assume they are in the order (the modifiable sections, unmodifiable sections) and the
        function will fuse them together.
        :param number_type: specifies the type of number that is being modified. Currently supports "int" or "float"
        """
        self.number_type = number_type.lower()
        for key in input_constraints:
            if key not in ["bounds", "eps"]:
                raise ValueError(key + " is an unrecognized constraint type")
            elif key == "bounds" and "method" not in input_constraints[key].keys():
                input_constraints[key]["method"] = 0
            elif key == "eps":
                if isinstance(input_constraints[key], int) or isinstance(
                    input_constraints[key], float
                ):  # If not provided, it is asumed to be abs
                    input_constraints[key] = {}
                    input_constraints[key]["value"] = temp_val
                    input_constraints[key]["type"] = "abs"
                elif "type" not in input_constraints[key].keys():
                    input_constraints[key]["type"] = "abs"
                elif "value" not in input_constraints[key].keys():
                    raise ValueError("Value must be provided for eps constraint")
                elif input_constraints[key]["type"] == "rel" and abs(input_constraints[key]["value"]) > 1:
                    raise ValueError("Illegal value. With relative eps, the value must be in the range (0,1]")

        input_constraints["eps"]["value"] = abs(input_constraints["eps"]["value"])

        for args in subtransformer_args:
            if "init_args" in args.keys():
                args["init_args"][
                    "number_type"
                ] = number_type  # Number type tells the subtransformers what data type they are modifying

        super(NumberTransformer, self).__init__(subtransformer_args, input_constraints, input_processor)

    def init_transformation_record(self, x):
        """
        Define the structure of the transformation record and initialize it.
        :param x: The input value to be transformed

        :return: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints
        """

        return 0

    def _enforce_constraints(self, x, transformation_record):
        """
        Enforce predefined input constraints
        :param x: Input value to be modified
        :param transformation_record: A record tracking the transformations already applied to x. For numbers, it is a int/float
        distance from the original value

        :return: A "clipped" input and corrected trasnformation record
        """

        original_value = x - transformation_record
        clipped_x = x
        if "eps" in self.input_constraints:
            # Absolute clipping - Clip based on a flat amount from the original value
            if self.input_constraints["eps"]["type"] == "abs":
                clipped_x = np.clip(
                    clipped_x,
                    original_value - self.input_constraints["eps"]["value"],
                    original_value + self.input_constraints["eps"]["value"],
                )
            # relative clipping - Clip based on a percentage of the original value
            else:
                clipped_x = np.clip(
                    clipped_x,
                    original_value - (original_value * self.input_constraints["eps"]["value"]),
                    original_value
                    + (self.input_constraints["bounds"].get("upper") * self.input_constraints["eps"]["value"]),
                )
        if "bounds" in self.input_constraints:
            method = self.input_constraints["bounds"]["method"]

            # For the following code, we assume the base features are within normal input bounds
            # 0: Always clip the values based on the mix/max bounds. (Default)
            # 1: Ensure that the value never drops below the base feature.
            # 2: Ensure that the value never goes above the base feature.
            # 3: Ensure that the value never drops below the base feature and the value is not clipped up to the lower bound.
            #    In other words, never clip up from the base value
            # 4: Ensure that the value never goes above the base feature and the value is not clipped down to the upper bound.
            #    In other words, never clip down from the base value
            # 5: A combination of 3 and 4. In other words, never clip in from the base value
            
            # 0
            lower_bound = self.input_constraints["bounds"].get("lower")
            upper_bound = self.input_constraints["bounds"].get("upper")

            if method == 1 or method == 3 or method == 5:
                lower_bound = original_value
                if (method == 3 or method == 5) and self.input_constraints["bounds"].get("lower") is not None:
                    lower_bound = min(lower_bound, self.input_constraints["bounds"].get("lower"))
                # Ensure that the value never goes above the base feature. -2 also means that ensure you don't clip down to the bound if the base feature is higher
            elif method == 2 or method == 4 or method == 5:
                upper_bound = original_value
                if (method == 4 or method == 5) and self.input_constraints["bounds"].get("upper") is not None:
                    upper_bound = max(upper_bound, self.input_constraints["bounds"].get("upper"))

            clipped_x = np.clip(clipped_x, lower_bound, upper_bound)
        corrected_transformation_record = transformation_record + (clipped_x - x)
        return clipped_x, corrected_transformation_record

    def _is_possible(self, x, transformation_record=None, transformation_value=None):
        """
        Determines if the action specified by transformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation
        :param transformation_value: The definition of the transformation to apply
        :return: True if possible, arguments to use for transform()
        """

        if transformation_value is None:
            return False, []  # Can't verify default actions

        possible, action_args, transformation_effect = self.subtransformer_list[transformation_value[0]].is_possible(
            x, *transformation_value[1:]
        )

        if not possible:
            return False, []

        # Validates the action with respect to the input constraints
        original_value = x - transformation_record
        # Action is invalid if the transformation record is at an eps boundary and the action's effect would move past it
        if "eps" in self.input_constraints:
            # Absolute clipping - Clip based on a flat amount from the original value.
            if self.input_constraints["eps"]["type"] == "abs":
                max_delta = self.input_constraints["eps"]["value"]
            # relative clipping - Clip based on a percentage of the original value.
            else:
                max_delta = (x - transformation_record) * self.input_constraints["eps"]["value"]

            if (transformation_effect > 0 and transformation_record >= max_delta) or (
                transformation_effect < 0 and transformation_record <= -1 * max_delta
            ):
                return False, []

        # Action is invalid if x is at an boundary for the range of valid values and the action's effect would move past it
        if "bounds" in self.input_constraints:
            method = self.input_constraints["bounds"]["method"]

            # For the following code, we assume the base features are within normal input bounds
            # 0: Always clip the values based on the mix/max bounds. (Default)
            # 1: Ensure that the value never drops below the base feature.
            # 2: Ensure that the value never goes above the base feature.
            # 3: Ensure that the value never drops below the base feature and the value is not clipped up to the lower bound.
            #    In other words, never clip up from the base value
            # 4: Ensure that the value never goes above the base feature and the value is not clipped down to the upper bound.
            #    In other words, never clip down from the base value
            # 5: A combination of 3 and 4. In other words, never clip in from the base value
            
            # 0
            lower_bound = self.input_constraints["bounds"].get("lower")
            upper_bound = self.input_constraints["bounds"].get("upper")
            
            if method == 1 or method == 3 or method == 5:
                lower_bound = original_value
                if (method == 3 or method == 5) and self.input_constraints["bounds"].get("lower") is not None:
                    lower_bound = min(lower_bound, self.input_constraints["bounds"].get("lower"))
                # Ensure that the value never goes above the base feature. -2 also means that ensure you don't clip down to the bound if the base feature is higher
            elif method == 2 or method == 4 or method == 5:
                upper_bound = original_value
                if (method == 4 or method == 5) and self.input_constraints["bounds"].get("upper") is not None:
                    upper_bound = max(upper_bound, self.input_constraints["bounds"].get("upper"))
                

            if (transformation_effect > 0 and x >= upper_bound) or (transformation_effect < 0 and x <= lower_bound):
                return False, []

        return True, action_args
