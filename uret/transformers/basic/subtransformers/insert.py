from uret.transformers import SubTransformer
import string

import numpy as np
from copy import deepcopy


class Insert(SubTransformer):

    name = "Insert"

    def __init__(self, transformation_range=None, subtransformer_index=None):
        """
        Initialize an `Insert` object. This object inserts a character into a string.
        :param transformation_range: The characters that can be inserted.
        :param subtransformer_index: The index of the subtransformer in the transformer list. If not none, the subtransformer assumes the transformation record 'actions_taken' value is a list.
        """
        if transformation_range is None:
            self.transformation_range = string.ascii_lowercase + string.digits
        else:
            self.transformation_range = transformation_range

        self.subtransformer_index = subtransformer_index
        super(Insert, self).__init__()

    def transform(self, x, transformation_record, transformation_value):
        """
        Inserts a value from to the input x
        :param x: Input text
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation
        :param transformation_value: Target character to insert

        :return: A transformed input
        :rtype:
        """

        if transformation_value is None:
            transformation_value = np.random.choice(self.transformation_range)
        elif transformation_value not in self.transformation_range:
            raise ValueError(transformation_value + " not in range")

        replacement_pos = np.random.randint(len(x))

        new_x = x[:replacement_pos] + transformation_value + x[replacement_pos:]

        transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
        if self.subtransformer_index is not None:
            transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
        else:
            transformation_record["current_state"]["actions_taken"] += 1

        transformation_record["current_state"]["delta"] = transformation_record["current_state"]["delta"] + len(
            transformation_value
        )
        transformation_record["current_state"]["value"] = new_x

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of characters that the subtransformer can insert
        """
        action_list = list(self.transformation_range)
        return action_list

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: True if possible, arguments to use for transform(), and the change in string length
        """
        if transformation_value not in self.transformation_range:
            return False, None, 0

        return True, [transformation_value], len(transformation_value)

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input. For string, only characters in the string range can be inserted
        :param x: Input Value

        :return: A list of transformation values
        """

        return self.get_action_list()
