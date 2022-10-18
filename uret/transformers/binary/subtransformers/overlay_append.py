from uret.transformers import SubTransformer
from uret.transformers.binary.subtransformers import binary_to_bytez

import random
import numpy as np

from copy import deepcopy


class OverlayAppend(SubTransformer):
    name = "OverlayAppend"

    def __init__(
        self, length=[6, 16], distribution_values=[0, 126, 255], max_actions=10, seed=None, subtransformer_index=None
    ):
        """
        Initialize a `OverlayAppend` object. This object adds bytes to the binary
        :param length: either an int representing the maximum append length or a pair of its represetning the (lower, upper) append
        length.
        :param distribution_values: A list of ints, which controls the possible byte values to be appended to the binary.
        :param max_actions: This determines how many actions are returned when actions are generated
        :param seed: Random seed value
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """
        if isinstance(length, list):
            self.min_val = length[0]
            self.max_val = length[1]
        else:
            self.min_val = 1
            self.max_val = length[1]

        self.distribution_values = distribution_values

        if max_actions <= 0:
            raise ValueError("max_actions must be greater than 0")

        self.max_actions = max_actions
        self.seed = seed
        self.subtransformer_index = subtransformer_index

        super(OverlayAppend, self).__init__()

    def transform(self, x, transformation_record, transformation_value, length=None):
        """
        Add bytes to the binary
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: Distribution to draw from when creating context
        :param length: length of content to append
        :return: A transformed input and modified transformation record
        """
        import lief # lgtm [py/repeated-import]

        random.seed(self.seed)

        if transformation_value is None:
            transformation_value = random.choice(self.distribution_values)
        if length is None:
            length = random.randint(self.min_val, self.max_val)

        # choose the upper bound for a uniform distribution in [0,upper]
        # upper chooses the upper bound on uniform distribution:
        # distribution_value =0 would append with all 0s
        # distribution_value =126 would append with "printable ascii"
        # distribution_value =255 would append with any character
        new_x = x + bytes([random.randint(0, transformation_value) for _ in range(length)])

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
        Return a list of (length, distribution_value) pairs. This will always return a list at least as long as the
        "distribution_value" list even if max actions is smaller.
        """

        action_list = list(self.distribution_values)

        return action_list

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """
        lengths = list(range(self.min_val, self.max_val + 1))

        if self.max_actions > 0:
            random.shuffle(lengths)
            lengths = lengths[: self.max_actions]

        return True, [[transformation_value, l] for l in lengths], None

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: input_value

        :return: A list of transformation values
        """

        return self.get_action_list()
