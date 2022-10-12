from uret.transformers import Transformer


class CategoryTransformer(Transformer):
    name = "CategoryTransformer"

    # Two attributes used by simulated annealing to track transformer properties
    transform_initial = True  # If True, then inputs to transform() should be based on the initial state of the sample

    def __init__(
        self,
        subtransformer_args,
        input_constraints={},
        input_processor=None,
    ):
        """
        Initialize a `CategoryTransformer` object.
        :param subtransformer_list: The functions to be used to transform the input of a certain data type.
        :param input_constraints: Not used
        :param input_processor: A function with two modes to control transforming input values where parts of the input should
        remain unchanged. If a single input is given, the function will return (the modifiable sections, unmodifiable sections).
        If two inputs are given, it is assume they are in the order (the modifiable sections, unmodifiable sections) and the
        function will fuse them together.
        """

        super(CategoryTransformer, self).__init__(subtransformer_args, input_constraints, input_processor)

    def init_transformation_record(self, x):
        """
        Define the structure of the transformation record and initialize it.
        :param x: The input value to be transformed

        :return:
        """

        return None

    def _enforce_constraints(self, x, transformation_record):
        """
        Not Used
        :param x: Input value to be modified
        :param transformation_record:

        :return: A "clipped" input and corrected transformation record
        """

        return x, transformation_record

    def _is_possible(self, x, transformation_record=None, transformation_value=None, *args, **kwargs):
        """
        Determines if the action specified by transformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_record: A record tracking the transformation already applied to x.
        :param transformation_value: The definition of the transformation to apply
        :return: True if possible, arguments to use for transform()
        """
        if transformation_value is None:
            return False, []  # Can't verify default actions

        possible, action_args, transformation_effect = self.subtransformer_list[transformation_value[0]].is_possible(
            x, *transformation_value[1:]
        )

        if not possible:
            return False, []

        return True, action_args
