from uret.transformers import SubTransformer
import random
import re
import string

from copy import deepcopy


class Delete(SubTransformer):

    name = "Delete"

    def __init__(self, transformation_range=None, subtransformer_index=None):
        """
        Initialize a `Delete` object. This object deletes a character from a string.
        :param transformation_range: The characters that can be deleted.
        :param subtransformer_index: The index of the subtrasformer in the transformer list. If not none, the subtransformer assumes the transformation record 'actions_taken' value is a list.
        """
        if transformation_range is None:
            self.transformation_range = string.ascii_lowercase + string.digits
        else:
            self.transformation_range = transformation_range

        self.subtransformer_index = subtransformer_index
        super(Delete, self).__init__()

    def transform(self, x, transformation_record, transformation_value, index=None):
        """
        Remove a value from to the input x
        :param x: Input text
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation
        :param transformation_value: Target character to remove
        :param index: Index of the character to remove

        :return: A transformed input
        :rtype:
        """

        if transformation_value is None:
            modifiable_values = [v for v in x if v in self.transformation_range]
            transformation_value = random.choice(modifiable_values)

        if index is None:
            replace_inds = [match.start() for match in re.finditer(transformation_value, x)]
            replacement_pos = random.choice(replace_inds)
            if len(replace_inds) == 0:
                print(transformation_value, "not present")
                return x, transformation_record

        elif x[index] == transformation_value:
            replacement_pos = index
        else:
            raise ValueError(transformation_value + " not at index")

        new_x = x[:replacement_pos] + x[replacement_pos + len(transformation_value) :]

        transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])

        if self.subtransformer_index is not None:
            transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
        else:
            transformation_record["current_state"]["actions_taken"] += 1

        transformation_record["current_state"]["delta"] = transformation_record["current_state"]["delta"] - 1 * len(
            transformation_value
        )
        transformation_record["current_state"]["value"] = new_x

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of characters that the subtransformer can delete
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

        replace_inds = [match.start() for match in re.finditer(transformation_value, x)]

        if len(replace_inds) == 0:
            return False, None, 0

        return True, [[transformation_value, ind] for ind in replace_inds], -1 * len(transformation_value)

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input. For string, only characters in the string can be deleted
        :param x: Input Value

        :return: A list of transformation values
        """

        uniq_chars = [c for c in list(set(x)) if c in self.transformation_range]

        return uniq_chars
