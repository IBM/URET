from uret.transformers import SubTransformer

import random
import os
import tempfile
import subprocess

import numpy as np
from copy import deepcopy


class UPXPack(SubTransformer):
    name = "UPXPack"

    def __init__(self, compression_levels=list(range(1, 10)), seed=None, subtransformer_index=None):
        """
        Initialize a `UPXPack` object. This object compresses the binary with UPX.
        :param compression_levels: An int or list of ints between [1,9] for the different UPX values.
        :param seed: Random seed value
        :param subtransformer_index: An int indicating where in the transformation record the subtransformer is defined. This is used to enforce "max # of transformations for this subtransformer" constraint.
        """

        if isinstance(compression_levels, int):
            self.compression_levels = [compression_levels]
        else:
            self.compression_levels = compression_levels

        if any(np.array(compression_levels) < 1) or any(np.array(compression_levels) > 9):
            raise ValueError("compression_levels must have values in the range [1,9]")

        self.seed = seed
        self.subtransformer_index = subtransformer_index

        super(UPXPack, self).__init__()

    def transform(self, x, transformation_record, transformation_value):
        """
        Pack x using UPX
        :param x: Input Value
        :param transformation_record: Record of changes
        :param transformation_value: The compression level
        :return: A transformed input and modified transformation record
        """

        random.seed(self.seed)
        tempfilename = os.path.join(tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

        # dump to a temporary file
        with open(tempfilename, "wb") as outfile:
            outfile.write(x)

        options = ["--force", "--overlay=copy"]
        options += ["-{}".format(transformation_value)]

        # optional arguments - We'll choose at random for simplicity
        options += ["--compress-exports={}".format(random.randint(0, 1))]
        options += ["--compress-icons={}".format(random.randint(0, 3))]
        options += ["--compress-resources={}".format(random.randint(0, 1))]
        options += ["--strip-relocs={}".format(random.randint(0, 1))]

        with open(os.devnull, "w") as DEVNULL:
            retcode = subprocess.call(
                ["upx"] + options + [tempfilename, "-o", tempfilename + "_packed"], stdout=DEVNULL, stderr=DEVNULL
            )

        os.unlink(tempfilename)

        # Op succcessful
        if retcode == 0:
            with open(tempfilename + "_packed", "rb") as infile:
                x = infile.read()
            os.unlink(tempfilename + "_packed")

            # update the transformation record
            # There is a chance it is none if we are just calling this function to see if UPX exists
            if transformation_record is not None:
                transformation_record["prev_state"] = deepcopy(transformation_record["current_state"])
                if self.subtransformer_index is not None:
                    transformation_record["current_state"]["actions_taken"][self.subtransformer_index] += 1
                else:
                    transformation_record["current_state"]["actions_taken"] += 1

                transformation_record["current_state"]["value"] = x

        return x, transformation_record

    def get_action_list(self):
        """
        Return  compression levels
        """
        return self.compression_levels

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
            self.transform(x, None, self.compression_levels[0])
            return self.compression_levels
        except FileNotFoundError:
            return []
