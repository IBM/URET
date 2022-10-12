from uret.transformers import SubTransformer

import lief
import random
import os
import tempfile
import subprocess

from copy import deepcopy


class UPXUnpack(SubTransformer):

    name = "UPXUnpack"

    def __init__(self, seed=None, subtransformer_index=None):
        """
        Initialize a `UPXUnpack` object. This object uses UPX to unpact the binary
        :param seed: The random seed used for transform
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """
        self.seed = seed
        self.subtransformer_index = subtransformer_index
        super(UPXUnpack, self).__init__()

    def transform(self, x, transformation_record, transformation_value=0):
        """
        Unpack x using UPX on input x
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: Not used
        :return: A transformed input and modified transformation record
        """
        random.seed(self.seed)
        tempfilename = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

        # dump to a temporary file
        with open(tempfilename, "wb") as outfile:
            outfile.write(x)

        with open(os.devnull, "w") as DEVNULL:
            retcode = subprocess.call(
                ["upx", tempfilename, "-d", "-o", tempfilename + "_unpacked"], stdout=DEVNULL, stderr=DEVNULL
            )

        os.unlink(tempfilename)
        # Op succcessful
        if retcode == 0:
            with open(tempfilename + "_unpacked", "rb") as result:
                x = result.read()
            os.unlink(tempfilename + "_unpacked")

            # update the transformation record
            transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
            if self.subtransformer_index is not None:
                transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
            else:
                transformation_record["current_state"]["actions_taken"] += 1

            transformation_record["current_state"]["value"] = x

        return x

    def get_action_list(self):
        """
        Return a default argument
        """
        return [0]  # Not Used

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by transformation_value can be performed on the current input.
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: If action is possible and the transformation value. The effect on the record is not used
        """
        # not sure how to check without actually packing
        try:
            self.transform(x, None, transformation_value)
            return True, [transformation_value], None
        except FileNotFoundError:
            return False, [], None  # not sure how to check without actually packing

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: input_value

        :return: A list of transformation values
        """
        # not sure how to check without actually packing
        try:
            self.transform(x, None, 0)
            return [0]
        except FileNotFoundError:
            return []  # not sure how to check without actually packing
