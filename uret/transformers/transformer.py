from abc import ABC, abstractmethod
import importlib  # This is used to import the subtransformers from a fixed directory
import random


class Transformer(ABC):
    """
    The abstract base class `Transformer` defines the basic requirements of a transformer
    """

    # Two attributes used by simulated annealing to track transformer properties
    transform_initial = False  # If True, then inputs to transform() should be based on the initial state of the sample
    transform_repeat = (
        False  # If True, then transform() should be called multiple times during one annealing move() call
    )

    def __init__(self, subtransformer_args, input_constraints={}, input_processor=None):
        """
        Initialize a `Transformer` object.
        :param subtransformer_args: Arguments used to initialize the subtransformers, which are functions that manipulate the
        input data. For example, if the Transformer was defined for numerical vlues, then a subtransformer might be a function
        that increments the value.
        :param input_constraints: The constraints to enforce on the input after transformation. For example, for numerical values
        these constraints might be clip bounds. The input constraints specific to the transformer will decide the structure of the
        'transformation_record'
        :param input_processor: A function with two modes to control transforming input values where parts of the input should
        remain unchanged. If a single input is given, the function will return (the modifiable sections, unmodifiable sections).
        If two inputs are given, it is assume they are in the order (the modifiable sections, unmodifiable sections) and the
        function will fuse them together. For example, maybe the input is a string "abc.com", but you only want to perform
        transformations on "abc"
        """
        self.subtransformer_list = []

        # The subtransformers should be stored in <current_module_path>.subtransformers for a particular task.
        subtransformer_directory = ".".join(self.__module__.split(".")[:-1])
        subtransformer_directory = subtransformer_directory + ".subtransformers"

        # DANGER DANGER BECAUSE USER DEFINE THE INPUT VALUE. LOOK AT LATER
        for args in subtransformer_args:
            subtransformer_module = importlib.import_module(subtransformer_directory)
            subtransformer = getattr(subtransformer_module, args["name"])
            if "init_args" in args.keys():
                self.subtransformer_list.append(subtransformer(**args["init_args"]))
            else:
                self.subtransformer_list.append(subtransformer())

        self.input_constraints = input_constraints
        self.input_processor = input_processor

    @abstractmethod
    def init_transformation_record(self, x):
        """
        Define the structure of the transformation record and initialize it.
        :param x: The input value to be transformed

        :return: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints
        """
        raise NotImplementedError

    def transform(self, x, transformation_record=None, transformation_value=None):
        """
        First it applies the input processing if defined, then performs the input transformation.
        :param x: The input value to be transformed
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints
        :param transformation_value: A class specific argument that defines the transformation to apply to x.
        This can be a single value or a tuple/list of values depending on the implementation.

        :return: A transformed input and the new transformation record
        """

        if self.input_processor:
            x, static_fields = self.input_processor(x)

        if transformation_record is None:
            transformation_record = self.init_transformation_record(x)

        # If nothing defined, randomly select a transformation function
        if transformation_value is None:
            transformation_index = random.choice(range(len(self.subtransformer_list)))
            x, transformation_record = self.subtransformer_list[transformation_index].transform(
                x, transformation_record, None
            )
        # Otherwise, use the specified transformation
        else:
            transformation_index = transformation_value[0]
            x, transformation_record = self.subtransformer_list[transformation_index].transform(
                x, transformation_record, *transformation_value[1:]
            )

        if len(self.input_constraints.keys()) != 0:
            x, transformation_record = self._enforce_constraints(x, transformation_record)

        if self.input_processor is not None:
            x = self.input_processor(x, static_fields)
        return x, transformation_record

    @abstractmethod
    def _enforce_constraints(self, x, transformation_record):
        """
        Enforce predefined input constraints
        :param x: Input value to be modified
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints

        :return: A "clipped" input and corrected trasnformation record
        """
        raise NotImplementedError

    def get_action_list(self):
        """
        Return a list of actions for the transformation specified based on the subtransformations
        """
        action_list = []
        for i in range(len(self.subtransformer_list)):
            subtransformer_action_list = self.subtransformer_list[i].get_action_list()
            for action in subtransformer_action_list:
                action_list.append([i, action])

        return action_list

    @abstractmethod
    def _is_possible(self, x, transformation_record=None, transformation_value=None):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints
        :param transformation_value: The definition of the transformation to apply

        :return: True if possible, arguments to use for transform()
        """
        raise NotImplementedError

    def is_possible(self, x, transformation_record=None, transformation_value=None):

        if transformation_value is None:
            return False, []

        if transformation_record is None:
            transformation_record = self.init_transformation_record(x)

        if self.input_processor:
            x, static_fields = self.input_processor(x)

        transformation_index = transformation_value[0]
        possible, action_args = self._is_possible(
            x, transformation_record=transformation_record, transformation_value=transformation_value
        )

        if not possible:
            return False, []

        # Check if the action args contain a single argument group or a list of multiple argument groups
        if not isinstance(action_args[0], list):
            return possible, [[transformation_index, *action_args]]
        else:
            return possible, [[transformation_index, *a] for a in action_args]

    def get_possible(self, x, transformation_record=None):
        """
        Determines what actions can be performed on the input.
        :param x: Input Value
        :param transformation_record: A record tracking the transformation already applied to x. This is used for constraint
        enforcement and action validation. The definition of the transformation record depends on the Transformer's possible input
        constraints

        :return: A list of transformation values and the subtransform indices
        """
        # We do this here so if_possible doesn't need to do it every time
        if transformation_record is None:
            transformation_record = self.init_transformation_record(x)

        action_list = []
        for i, subtransformer in enumerate(self.subtransformer_list):
            subtransformer_actions = subtransformer.get_possible(x)
            for action in subtransformer_actions:
                possible, action_args = self.is_possible(
                    x, transformation_record=transformation_record, transformation_value=[i, action]
                )
                if possible:
                    for a in action_args:
                        action_list.append(a)

        return action_list
