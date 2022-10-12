from uret.transformers import Transformer
import numpy as np


class BinaryTransformer(Transformer):
    name = "BinaryTransformer"

    def __init__(self, subtransformer_args, input_constraints={}, input_processer=None):
        """
        Initialize a `BinaryTransformer` object.
        :param subtransformer_list: The functions to be used to transform the input of a certain data type.
        :param input_constraints: The constraints to enforce on the input after transformation. The possible enforcement values
        are:
            - max_actions: An int representing the maximum number of subtransformer actions allowed to be performed.
            - max_subtransformer_actions: A optional list of values with length equal to the number of subtransformers. Each element represents the max number of times the corresponding subtransformer can be called.
            relative constraint with respect to the original length of the string.
        :param input_processer: A function with two modes to control transforming input values where parts of the input should
        remain unchanged. If a single input is given, the function will return (the modifiable sections, unmodifiable sections).
        If two inputs are given, it is assume they are in the order (the modifiable sections, unmodifiable sections) and the
        function will fuse them together.
        """

        for key in input_constraints:
            if key not in ["max_actions", "max_subtransformer_actions"]:
                raise ValueError(key + " is an unrecognized constraint type")
            elif key == "max_subtransformer_actions":
                if isinstance(input_constraints[key], list):
                    if any(np.array(input_constraints[key])) <= 1:
                        raise ValueError(key + " cannot be <= 1")
                    if len(subtransformer_args) != len(input_constraints[key]):
                        raise ValueError(key + " must be the same length as the number of subtransformers if a list")
                    # This is so the subtransformer will know the output
                    # format of the transformation record
                    for i, args in enumerate(subtransformer_args):
                        if "init_args" not in args.keys():
                            args["init_args"] = {}
                        args["init_args"]["subtransformer_index"] = i
            if input_constraints[key] <= 1:
                raise ValueError(key + " cannot be <= 1")

        super(BinaryTransformer, self).__init__(subtransformer_args, input_constraints, input_processer)

    def init_transformation_record(self, x):
        """
        Define the structure of the transformation record and initialize it.
        :param x: The input value to be transformed

        :return: A record that records the previous and current binary state so we can undo changes if needed. We only look 1
        state back because the prev state is guaranteed to be valid and it is the next state that must be verified
        """

        actions_taken = 0
        if "max_actions" in self.input_constraints:
            if self.input_constraints["max_actions"] < 1:
                max_actions = np.ceil(len(x) * self.input_constraints["max_actions"])
            else:
                max_actions = int(self.input_constraints["max_actions"])
        else:
            max_actions = None
            
        if "max_subtransformer_actions" in self.input_constraints:
                max_subtransformer_actions = []
                actions_taken = []
                for v in self.input_constraints["max_subtransformer_actions"]:
                    if v < 1:
                        max_subtransformer_actions.append(len(x) * v)
                    else:
                        max_subtransformer_actions.append(int(v))
                    actions_taken.append(0)
        else:
            max_subtransformer_actions = None

        return {
            "prev_state": {"actions_taken": actions_taken, "value": x},
            "current_state": {"actions_taken": actions_taken, "value": x},
            "max_subtransformer_actions": max_subtransformer_actions,
            "max_actions": max_actions,
        }

    def _enforce_constraints(self, x, transformation_record):
        """
        Enforce predefined input constraints
        :param x: Input value to be modified
        :param transformation_record: A record tracking the transformations already applied to x.

        :return: A "clipped" input and corrected transformation record. If clipping is done, the function "rewinds" back to the
        previous state
        """

        clipped_x = x
        corrected_transformation_record = transformation_record

        if "max_actions" in self.input_constraints:
            if (isinstance(transformation_record["current_state"]["actions_taken"], list) and np.sum(transformation_record["current_state"]["actions_taken"]) > transformation_record["max_actions"]) or (isinstance(transformation_record["current_state"]["actions_taken"], int) and transformation_record["current_state"]["actions_taken"] > transformation_record["max_actions"]): 
                clipped_x = transformation_record["prev_state"]["value"]  # The string prior to transformation
                corrected_transformation_record["current_state"] = corrected_transformation_record["prev_state"].copy()

        if "max_subtransformer_actions" in self.input_constraints and any(
                np.array(transformation_record["current_state"]["actions_taken"])
                > np.array(transformation_record["max_subtransformer_actions"])
            ):
                clipped_x = corrected_transformation_record["prev_state"]["value"]  # The string prior to transformation
                corrected_transformation_record["current_state"] = corrected_transformation_record["prev_state"].copy()

        return clipped_x, corrected_transformation_record

    def _is_possible(self, x, transformation_record=None, transformation_value=None):
        """
        Determines if the action specified by tranformation_value can be performed on the current input. If it can, it returns
        True and provides a list of the arguments to use with transform().
        :param x: Input value
        :param transformation_value: The definition of the transformation to apply and the parameters to use
        :return: True if possible, list of possible arguments
        """
        if transformation_value is None:
            return False, []  # Can't verify default actions

        # Check action constraint. This can be done without consulting the subtransformer
        if "max_subtransformer_actions" in self.input_constraints:
            actions_taken = transformation_record["current_state"]["actions_taken"]
            next_action = np.zeros_like(actions_taken)
            next_action[transformation_value[0]] += 1
            predicted_taken = np.array(actions_taken) + next_action
            if any(predicted_taken > np.array(transformation_record["max_subtransformer_actions"])) or ("max_actions" in self.input_constraints and np.sum(predicted_taken) > transformation_record["max_actions"]):
                return False, []
        elif "max_actions" in self.input_constraints and transformation_record["current_state"]["actions_taken"] + 1 > transformation_record["max_actions"]:
                return False, []

        # Ask the subtransformer if it is possible
        possible, action_args, _ = self.subtransformer_list[transformation_value[0]].is_possible(
            x, *transformation_value[1:]
        )

        if not possible:
            return False, []

        return possible, action_args
