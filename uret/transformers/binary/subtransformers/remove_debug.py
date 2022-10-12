from uret.transformers import SubTransformer
from uret.transformers.binary.subtransformers import binary_to_bytez

import lief
from copy import deepcopy


class RemoveDebug(SubTransformer):
    name = "RemoveDebug"

    directory_mapper = {
        "has_signature": lief.PE.DATA_DIRECTORY.CERTIFICATE_TABLE,
        "has_debug": lief.PE.DATA_DIRECTORY.DEBUG,
    }

    def __init__(self, sub=None):
        """
        Initialize a `RemoveDebug` object. This object removes the debug data directory .
        :param sub: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """
        self.sub = sub

        super(RemoveDebug, self).__init__()

    def transform(self, x, transformation_record, transformation_value=0):
        """
        Removed the specified data directory
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: Not used
        :return: A transformed input and modified transformation record
        """

        binary = lief.PE.parse(list(x))

        for i, e in enumerate(binary.data_directories):
            if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
                break

        if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
            e.rva = 0
            e.size = 0

            new_x = binary_to_bytez(binary)

            # update the transformation record
            transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
            if self.sub is not None:
                transformation_record["current_state"]["actions_taken"][self.sub] += 1
            else:
                transformation_record["current_state"]["actions_taken"] += 1

            transformation_record["current_state"]["value"] = new_x

            return new_x, transformation_record

        else:
            return x, transformation_record

    def get_action_list(self):
        """
        Returns a default argument
        """
        return [0]

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """

        binary = lief.PE.parse(list(x))

        for i, e in enumerate(binary.data_directories):
            if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
                return True, [transformation_value], None

        return False, [], None

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: input_value

        :return: A list of transformation values
        """
        binary = lief.PE.parse(list(x))

        for i, e in enumerate(binary.data_directories):
            if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
                return [0]

        return []
