from uret.transformers import SubTransformer
import warnings

import numpy as np


class CategoryModifier(SubTransformer):

    name = "CategoryModifier"

    def __init__(self, categories, is_onehot=False):
        """
        Initialize an `CategoryModifier` object.
        :param categories: A list possible feature values if not a one-hot feature. If the feature is a one-hot feature, then this  is ignored.
        :param is_onehot: Indicates if the feature is a single category featuer or a one-hot (ie multi category) feature
        """

        self.categories = categories
        self.is_onehot = is_onehot

        super(CategoryModifier, self).__init__()

    def transform(self, x, transformation_record, transformation_value):
        """
        :param x: The input value to be transformed. If a one-hot feature, then this contains all of the values
        :param transformation_record: Not used for enforcement
        :param transformation_value: Either the category to switch or the index of the category to set hot
        This can be a single value or a tuple/list of values depending on the implementation.

        :return: A transformed input and the new transformation record
        """

        if self.is_onehot and len(x) != self.categories:
            raise RuntimeError(
                f"Onehot feature of length {len(x)}, but categories is set to {self.categories}. These should be equal for onehot features"
            )

        if transformation_value is None:
            possible_values = self.get_possible()
            transformation_value = random.choice(possible_values)

        if self.is_onehot:
            x[x == 1] = 0
            x[transformation_value] = 1
        else:
            x = transformation_value
            
        transformation_record = transformation_value

        return x, transformation_record

    def get_action_list(self):
        """
        Return a list of transformation values
        """
        if self.is_onehot:
            return list(set(np.arange(self.categories)))

        return list(set(self.categories))

    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The index to transform (one-hot) or category to change to

        :return: True if possible, arguments to use for transform()
        """

        if self.is_onehot and x[transformation_value] == 1:
            return False, [], None
        elif not self.is_onehot and x == transformation_value:
            return False, [], None
        return True, [transformation_value], transformation_value

    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: Input Value

        :return: A list of transformation values
        """

        if self.is_onehot:
            if len(x) != self.categories:
                raise RuntimeError(
                    f"Onehot feature of length {len(x)}, but categories is set to {self.categories}. These should be equal for onehot features"
                )
            possible_values = list(set(np.arange(self.categories)) - set(np.where(x == 1)[0]))
        else:
            possible_values = list(set(self.categories) - set(list(x)))

        return possible_values
