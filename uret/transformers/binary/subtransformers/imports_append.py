from uret.transformers import SubTransformer
from uret.transformers.binary.subtransformers import binary_to_bytez

import random
import os

import json
from copy import deepcopy


class ImportsAppend(SubTransformer):
    name = "ImportsAppend"

    def __init__(self, transformation_range=None, max_actions=50, seed=None, subtransformer_index=None):
        """
        Initialize a `ImportsAppend` object. This object adds a new function (and library) to the import list
        :param transformation_range: A dictionary of (libary_name, function_list) pairs. If not provided, a dictionary of common
        library imports will be used
        :param max_actions: If > 0, this determines how many actions are returned when get_action_list is called
        :param seed: Random seed value
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """

        if transformation_range is None:
            self.transformation_range = json.load(
                open("agrex/transformers/binary/subtransformers/small_dll_imports.json", "r")
            )
        elif not isinstance(transformation_range, dict):
            raise ValueError("transformation_range must be a dictionary")
        else:
            self.transformation_range = transformation_range

        if max_actions <= 0:
            raise ValueError("max_actions must be greater than 0")

        self.max_actions = max_actions
        self.seed = seed
        self.subtransformer_index = subtransformer_index

        super(ImportsAppend, self).__init__()

    def transform(self, x, transformation_record, transformation_value):
        """
        Add a new function from the library specified by the transformation value to the import list
        :param x: Input Value. bytes
        :param transformation_record: Record of changes
        :param transformation_value: libary to append
        :return: A transformed input and modified transformation record
        """
        import lief # lgtm [py/repeated-import]

        random.seed(self.seed)

        binary = lief.PE.parse(list(x))

        if transformation_value is None:
            transformation_value = random.choice(list(self.transformation_range.keys()))

        # Find the library if it already exists in the import list
        lib = None
        for im in binary.imports:
            if im.name.lower() == transformation_value.lower():
                lib = im
                break

        # If not, add it
        if lib is None:
            lib = binary.add_library(transformation_value)

        # Then add a random function that doesn't exist
        possible_values = list(
            set(self.transformation_range[transformation_value]) - set([e.name for e in lib.entries])
        )
        func_to_add = random.choice(possible_values)
        lib.add_entry(func_to_add)

        new_x = binary_to_bytez(binary, imports=True)

        # update the transformation record
        transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
        if self.subtransformer_index is not None:
            transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
        else:
            transformation_record["current_state"]["actions_taken"] += 1

        transformation_record["current_state"]["value"] = new_x

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of possible library imports
        """
        action_list = list(self.transformation_range.keys())
        if self.max_actions > 0:
            random.shuffle(action_list)
            return action_list[: self.max_actions]
        return action_list

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """
        import lief

        binary = lief.PE.parse(list(x))

        # Find the library if it already exists in the import list
        lib = None
        for im in binary.imports:
            if im.name.lower() == transformation_value.lower():
                lib = im
                break

        # If library is not in the list, we can definitely append
        if lib is None:
            return True, [[transformation_value]], None

        # Can't append library if there are no functions left to append
        possible_values = list(
            set(self.transformation_range[transformation_value]) - set([e.name for e in lib.entries])
        )
        if len(possible_values) == 0:
            return False, None, None

        return True, [transformation_value], None

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input. It randomly generates the action list and then filters out any
        libraries that are completly imported
        :param x: input_value

        :return: A list of transformation values
        """
        all_actions = self.get_action_list()
        possible_actions = []
        for action in all_actions:
            is_possible, _, _ = self.is_possible(x, action)
            if is_possible:
                possible_actions.append(action)

        return possible_actions
