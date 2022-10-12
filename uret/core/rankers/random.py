from uret.core.rankers.ranking_algorithm import RankingAlgorithm

import numpy as np
import random
import copy
import warnings


class Random(RankingAlgorithm):
    """
    This implementation tries a random transformation for given sample.
    """

    def __init__(self, transformer_list, multi_feature_input=False, is_trained=True, num_actions=1):
        """
        :param transformer_list: List of available transformations to be used by the algorithm.
        :param multi_feature_input: A boolean flag indicating if each index in the transformer
            list defines a transformer(s) for a single data type in the input. If False, then it
            is assumed that the transformer list defines multiple transformers for a single input
        :param is_trained: Indicates when _train has been called at least once. Always True
        :param num_actions: Number of random actions to sample
        """
        super().__init__(transformer_list, multi_feature_input, is_trained=True)

        if num_actions < 1:
            num_actions = 1
        self.num_actions = num_actions

    def rank_edges(
        self,
        sample,
        scoring_function,
        score_input,
        dependencies=[],
        current_transformation_records=None,
        transformer_index=None,
    ):
        """
        param transformer_index: Index of the transformer to use for this ranking. This is a parameter used by annealing to randomly select an action for a specific transformer
        """

        # Create transformation record
        if self.multi_feature_input and current_transformation_records is None:
            current_transformation_records = [None for _ in range(len(self.transformer_list))]

        sample_temp = copy.copy(sample)
        transformation_records_temp = copy.copy(current_transformation_records)

        # Get a random transformer from the list. Make sure there is at least one possible action to perform
        # Exception is when annealing provides the specific index to us
        possible_actions = []
        if transformer_index is None:
            skip_inds = []
            while len(possible_actions) == 0:
                if len(skip_inds) == len(self.transformer_list):
                    return []  # No actions are possible

                transformer_index = random.choice(list(set(np.arange(len(self.transformer_list))) - set(skip_inds)))
                transformer, input_index = self.transformer_list[transformer_index]
                if self.multi_feature_input:
                    possible_actions = transformer.get_possible(
                        sample_temp[input_index], transformation_record=transformation_records_temp[transformer_index]
                    )
                else:
                    possible_actions = transformer.get_possible(
                        sample_temp, transformation_record=transformation_records_temp
                    )

                skip_inds.append(transformer_index)
        else:
            transformer, input_index = self.transformer_list[transformer_index]
            if self.multi_feature_input:
                possible_actions = transformer.get_possible(
                    sample_temp[input_index], transformation_record=transformation_records_temp[transformer_index]
                )
            else:
                possible_actions = transformer.get_possible(
                    sample_temp, transformation_record=transformation_records_temp
                )

            if len(possible_actions) == 0:
                return []  # No actions are possible

        return_values = []  # This will contain the (sample_index, transformer, action args, transformed sample,
        # transformation_record of the transformed sample, score)

        # Pick a random action  to perform
        actions = random.sample(possible_actions, k=min(len(possible_actions), self.num_actions))

        for action in actions:
            sample_temp = copy.copy(sample)
            transformation_records_temp = copy.copy(current_transformation_records)

            if self.multi_feature_input:
                transformed_value, new_transformation_record = transformer.transform(
                    sample_temp[input_index],
                    transformation_record=transformation_records_temp[transformer_index],
                    transformation_value=action,
                )
                sample_temp[input_index] = transformed_value
                transformation_records_temp[transformer_index] = new_transformation_record
            else:
                transformed_value, new_transformation_record = transformer.transform(
                    sample_temp, transformation_record=transformation_records_temp, transformation_value=action
                )
                sample_temp = transformed_value
                transformation_records_temp = new_transformation_record

            sample_temp = self._enforce_dependencies(sample_temp, dependencies)
            score = scoring_function(sample_temp, score_input)

            if self.multi_feature_input:
                return_values.append(
                    (
                        [transformer_index, input_index],
                        transformer,
                        action,
                        sample_temp,
                        transformation_records_temp,
                        score,
                    )
                )
            else:
                return_values.append((None, transformer, action, sample_temp, transformation_records_temp, score))

        return return_values
