from abc import ABC, abstractmethod


class SubTransformer(ABC):
    """
    The abstract base class `SubTransformer` defines the basic requirements of a SubTransformer. A SubTransformer is a class that manipulates a specified data type.
    """

    def __init__(self):
        """
        Initialize a `SubTransformer` object.
        """
        pass

    @abstractmethod
    def transform(self, x, transformation_record, transformation_value=None, *args, **kwargs):
        """
        Public method. First it applies the input processing if defined, then performs the input transformation.
        :param x: The input value to be transformed
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation
        :param transformation_value: The definition of the transformation to apply

        :return: A transformed input and the new transformation record
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_list(self):
        """
        Return a list of possible actions for the transformation
        """
        raise NotImplementedError

    @abstractmethod
    def is_possible(self, x, transformation_value):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply

        :return: True if possible, arguments to use for transform(), and effect on the transformation record (from Transformer)
        with respect to the Transformer's possible input constraints
        """
        raise NotImplementedError

    @abstractmethod
    def get_possible(self, x):
        """
        Determines what actions can be performed on the input.
        :param x: Input Value

        :return: A list of transformation values
        """
        raise NotImplementedError
