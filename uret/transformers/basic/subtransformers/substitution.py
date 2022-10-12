from uret.transformers import SubTransformer

import random
import re
import string

from copy import deepcopy


class Substitution(SubTransformer):
    name = "Substitution"

    def __init__(self, transformation_range=None, max_actions=0, subtransformer_index=None):
        """
        Initialize a `Substitution` object. This object substitutes a character in a string.
        :param transformation_range: The characters that can be deleted.
        :param max_actions: If > 0, this determines how many actions are returned when get_action_list is called
        :param subtransformer_index: The index of the subtransformer in the transformer list. If not none, the subtransformer assumes the transformation record 'actions_taken' value is a list.
        """
        if transformation_range is None:
            self.transformation_range = string.ascii_lowercase + string.digits
        else:
            self.transformation_range = transformation_range

        self.max_actions = max_actions
        self.subtransformer_index = subtransformer_index

        super(Substitution, self).__init__()

    def transform(self, x, transformation_record, transformation_value, index=None):
        """
        Perform a substitution on input x based on the remaining input arguments
        :param x: Input text
        :param transformation_value: (Target value, new value) pair
        :param index: Index of the character to make the substitution. If not provied, a random matching position will be used
        :return: A transformed input and the modified record
        """

        if transformation_value is not None:
            target_value, new_value = transformation_value
        else:
            target_value = random.choice([v for v in x if v in self.transformation_range])
            new_value = None

        if target_value not in self.transformation_range or (
            new_value is not None and new_value not in self.transformation_range
        ):
            raise ValueError("Transformation value is not in the pre-defined range")

        if new_value is None:
            new_value = random.choice(self.transformation_range.replace(target_value, ""))

        if index is None:
            replace_inds = [match.start() for match in re.finditer(target_value, x)]

            if len(replace_inds) == 0:
                print(target_value, "not present")
                return x
            replacement_pos = random.choice(replace_inds)

        elif x[index : index + len(new_value)] == target_value:
            replacement_pos = index

        else:
            raise ValueError(transformation_value + " wasn't found at the specified index")

        new_x = x[:replacement_pos] + new_value + x[replacement_pos + len(new_value) :]

        transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
        if self.subtransformer_index is not None:
            transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
        else:
            transformation_record["current_state"]["actions_taken"] += 1

        transformation_record["current_state"]["value"] = new_x

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of characters pairs that the subtransformer can replace
        """
        action_list = [
            (old, new) for new in self.transformation_range for old in self.transformation_range if old != new
        ]

        if self.max_actions > 0:
            action_list.shuffle()
            return action_list[: self.max_actions]
        return action_list

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: True if possible, arguments to use for transform(), and the change in string length
        """
        target_value, new_value = transformation_value
        if target_value not in self.transformation_range or (
            new_value is not None and new_value not in self.transformation_range
        ):
            return False, None, 0

        replace_inds = [match.start() for match in re.finditer(target_value, x)]

        if len(replace_inds) == 0:
            return False, None, 0

        return True, [[transformation_value, ind] for ind in replace_inds], 0

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input. For string, only characters in the string can be replaced
        :param x: Input Value

        :return: A list of transformation values
        """
        uniq_chars = [c for c in list(set(x)) if c in self.transformation_range]
        possible_actions = [(old, new) for new in self.transformation_range for old in uniq_chars if old != new]

        return possible_actions
