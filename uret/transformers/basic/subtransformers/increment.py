from uret.transformers import SubTransformer
import warnings

import numpy as np


class Increment(SubTransformer):

    name = "Increment"

    def __init__(self, low=0, high=1, number_type="float", action_samples=10, sample_method="random"):
        """
        Initialize an `Increment` object. This object increments an object by a numerical amount depending on the hyper
        parameters.
        :param low: The lower bound (inclusive) to sample increment values from. It should be a non-negative number.
        :param high: The upper bound (exclusive) to sample increment values from. It should be a non-negative number.
        :param number_type: specifies the type of number that is being modified. Currently supports "int" or "float"
        :param sample_num: Max number of samples generated when get_action_list() is called.
        :param sample_method: Determines how action samples are generated when get_action_list() is called
            - random: Generate samples at random within the range (low, high) with a uniform probability (Default)
            - linspace: Generate sample using linearly spaced samples within the range (low, high)
            - geomspace: Generate sample using log spaced samples within the range (low, high)
        """

        if abs(high) <= abs(low):
            raise ValueError("high must be greater than low")

        self.low = abs(low)
        self.high = abs(high)

        self.number_type = number_type
        if number_type == "int":
            if high - low < int(action_samples / 2):  # Corner case so the number of samples isn't more than the range
                self.action_samples = 2 * (high - low)
            else:
                self.action_samples = action_samples
        else:
            self.action_samples = action_samples
        
        if sample_method.lower() not in ["random", "linspace", "geomspace"]:
            raise ValueError("Sample method not supported. Supported methods are random,linspace, and geomspace")
        self.sample_method = sample_method.lower()

        super(Increment, self).__init__()

    def transform(
        self,
        x,
        transformation_record,
        transformation_value,
    ):
        """
        Public method. First it applies the input processing if defined, then performs the input transformation.
        :param x: The input value to be transformed
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation
        :param transformation_value: A class specific argument that defines the transformation to apply to x.
        This can be a single value or a tuple/list of values depending on the implementation.

        :return: A transformed input and the new transformation record
        """

        if transformation_value is None:
            multiplier = 1 if np.random.random() < 0.5 else -1

            if self.number_type == "int":
                increment_value = multiplier * np.random.randint(self.low, self.high)
            elif self.number_type == "float":
                increment_value = multiplier * np.random.uniform(self.low, self.high)

        else:
            increment_value = transformation_value

        new_x = x + increment_value
        transformation_record = transformation_record + increment_value

        return new_x, transformation_record

    def get_action_list(self):
        """
        Return a list of possible values to increment by
        """
        
        sample_num = int(self.action_samples / 2)  # We divide by two cause we will be using pos and negative values

        low = self.low
        high = self.high

        if self.sample_method == "random":
            if self.number_type == "int":
                increment_values = np.random.randint(low, high, sample_num)
            elif self.number_type == "float":
                increment_values = np.random.uniform(low, high, sample_num)

        elif self.sample_method == "linspace":
            if self.number_type == "int":
                increment_values = np.linspace(low, high, sample_num, endpoint=False, dtype=int)
            elif self.number_type == "float":
                increment_values = np.linspace(low, high, sample_num, endpoint=False)

        elif self.sample_method == "geomspace":
            if low == 0:
                low = 0.000001
            if self.number_type == "int":
                increment_values = np.geomspace(low, high, sample_num, endpoint=False, dtype=int)
            elif self.number_type == "float":
                increment_values = np.geomspace(low, high, sample_num, endpoint=False)

        increment_values = np.array(
            list(set(increment_values))
        )  # remove duplicates. This can be from geomspace with ints usually
        increment_values = np.delete(increment_values, increment_values == 0)  # remove 0's
        
        increment_values = np.concatenate((-1 * increment_values, increment_values))  # Use positive and negative values

        return increment_values

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: True if possible, arguments to use for transform(), and the change in delta
        """

        return True, [transformation_value], transformation_value

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input. For numbers, anything is possible because the subtransformer
        doesn't need to check constraints
        :param x: Input Value

        :return: A list of transformation values
        """

        return self.get_action_list()
