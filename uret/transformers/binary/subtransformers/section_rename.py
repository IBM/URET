from uret.transformers import SubTransformer
from uret.transformers.binary.subtransformers import binary_to_bytez

import random
from copy import deepcopy


class SectionRename(SubTransformer):
    name = "SectionRename"

    def __init__(self, section_names=None, seed=None, subtransformer_index=None):
        """
        Initialize a `SectionRename` object. This object renames a section.
        :param section_names: A list of strings containing section names
        :param seed: Random seed value
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """
        self.seed = seed

        if section_names is None:
            self.section_names = (
                open("agrex/transformers/binary/subtransformers/section_names.txt", "r").read().rstrip().split("\n")
            )

        self.seed = seed
        self.subtransformer_index = subtransformer_index

        super(SectionRename, self).__init__()

    def transform(self, x, transformation_record, transformation_value=None):
        """
        Add bytes to the binary
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: (length, distribution) to append
        :return: A transformed input and modified transformation record
        """
        import lief # lgtm [py/repeated-import]

        random.seed(self.seed)
        binary = lief.PE.parse(list(x))

        if transformation_value is None:
            transformation_value = random.choice(self.section_names)

        # Pick a random section to rename
        target_section = random.choice(binary.sections)
        if target_section.name != transformation_value:
            target_section.name = transformation_value
            new_x = binary_to_bytez(binary)

            # update the transformation record
            transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
            if self.subtransformer_index is not None:
                transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
            else:
                transformation_record["current_state"]["actions_taken"] += 1

            transformation_record["current_state"]["value"] = new_x
            return new_x, transformation_record
        else:
            return x, transformation_record

    def get_action_list(self):
        """
        Return a list of section names
        """
        return self.section_names

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """
        return True, [transformation_value], None

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: input_value

        :return: A list of transformation values
        """
        return self.section_names
