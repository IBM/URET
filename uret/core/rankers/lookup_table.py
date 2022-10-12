from uret.core.rankers.ranking_algorithm import RankingAlgorithm
from uret.core.rankers.brute_force import BruteForce

import warnings
import tqdm
import scipy.spatial
import numpy


class LookupTable(RankingAlgorithm):
    """
    This implementation uses a pre-computed lookup table. It uses the brute_force method on the training samples to built a lookup table
    """

    def __init__(self, transformer_list, multi_feature_input=False, is_trained=False, feature_distance_type="sim"):
        """
        :param transformer_list: List of available transformations to be used by the algorithm.
        :param multi_feature_input: A boolean flag indicating if each index in the transformer
            list defines a transformer(s) for a single data type in the input. If False, then it
            is assumed that the transformer list defines multiple transformers for a single input
        :param is_trained: Indicates when _train has been called at least once
        :param feature_distance_type: Function to use when determining which action to use in the lookup table if table values are
            feature perturbation vectors. Option are
                sim - Cosine similarity between perturbation vectors
                l2 - l2 distance
        """
        super().__init__(transformer_list, multi_feature_input, is_trained=False)

        self.brute_force = BruteForce(transformer_list, multi_feature_input=multi_feature_input)
        self.lookup_table = {}

        if feature_distance_type != "sim" and feature_distance_type != "l2":
            raise ValueError(f"Distance type not supported yet")

        self.feature_distance_type = feature_distance_type

    def _train(
        self,
        training_samples,
        value_function,
        value_input_function,
        scoring_alg,
        dependencies=[],
        update=False,
    ):
        """
        Create a lookup table using the training samples. We run the brute force algorithm to get scores for the training data with respect to the value_function.
        :param training_samples: Input samples used to train the action dictionary
        :param value_function: Function used to compute values to compute the value a transformation has on the input with respect to a certain loss.
        :param value_input_function: Function used to compute the initial state of the sample before transform. This is fed into the value function as an input.
        :param scoring_alg: Indicates the scoring alg used for ranking if necessary
        :param dependencies: Input dependencies that need to be enforced
        """

        if not update:
            self.lookup_table = {}

        self.scoring_alg = scoring_alg  # This informs the lookup table how to compute during edge ranking
        self.value_input_function = value_input_function  # This enables re-use during edge ranking

        # Use all of the training samples and run the brute force algorithm
        print(f"Creating Dictionary")
        for i, sample in tqdm.tqdm(enumerate(training_samples)):
            initial_state = value_input_function(sample)
            edge_transform_estimates = self.brute_force.rank_edges(
                sample,
                value_function,
                initial_state,
                dependencies=dependencies,
                current_transformation_records=None,
            )

            for indices, transformer, action, _, _, score in edge_transform_estimates:
                action_key = str(action)  # Need to convert from list to hashable type
                if self.multi_feature_input:
                    if indices[0] not in self.lookup_table.keys():
                        self.lookup_table[indices[0]] = {"input_index": indices[1]}
                    if transformer.name not in self.lookup_table[indices[0]].keys():
                        self.lookup_table[indices[0]][transformer.name] = {}
                    if action_key not in self.lookup_table[indices[0]][transformer.name].keys():
                        self.lookup_table[indices[0]][transformer.name][action_key] = {"avg_change": 0, "count": 0}

                    self.lookup_table[indices[0]][transformer.name][action_key]["count"] += 1
                    self.lookup_table[indices[0]][transformer.name][action_key]["avg_change"] += (
                        score - self.lookup_table[indices[0]][transformer.name][action_key]["avg_change"]
                    ) / self.lookup_table[indices[0]][transformer.name][action_key]["count"]

                else:
                    if transformer.name not in self.lookup_table.keys():
                        self.lookup_table[transformer.name] = {}
                    if action_key not in self.lookup_table[transformer.name].keys():
                        self.lookup_table[transformer.name][action_key] = {"avg_change": 0, "count": 0}

                    self.lookup_table[transformer.name][action_key]["count"] += 1
                    self.lookup_table[transformer.name][action_key]["avg_change"] += (
                        score - self.lookup_table[transformer.name][action_key]["avg_change"]
                    ) / self.lookup_table[transformer.name][action_key]["count"]

        print(f"Dictionary Created")
        self.is_trained = True

    def rank_edges(self, sample, scoring_function, score_input, dependencies=[], current_transformation_records=None):

        if not self.is_trained:
            raise RuntimeError(
                f"Lookup table has not been trained. Please call `GraphExplorer.train(training_samples)` first"
            )

        # Create tranformation record
        if self.multi_feature_input and current_transformation_records is None:
            current_transformation_records = [None for _ in range(len(self.transformer_list))]

        return_values = []  # This will contain the (sample_index, transformer, action args, transformed sample,
        # transformation_record of the transformed sample, score)

        for transformer_index, (transformer, input_index) in enumerate(self.transformer_list):
            # Get the possible actions
            if self.multi_feature_input:
                possible_actions = transformer.get_possible(
                    sample[input_index], transformation_record=current_transformation_records[transformer_index]
                )
                action_dict = self.lookup_table[transformer_index][transformer.name]
            else:
                possible_actions = transformer.get_possible(
                    sample, transformation_record=current_transformation_records
                )
                action_dict = self.lookup_table[transformer.name]

            for action in possible_actions:
                action_key = str(action)
                if action_key not in action_dict.keys():
                    continue

                predicted_perturbation = action_dict[action_key]["avg_change"]
                if self.scoring_alg == "model_loss":
                    estimated_score = predicted_perturbation  # In the case of probabilities, we can just look for the largest positive value
                # If it's feature loss we compute the specified metric
                elif self.feature_distance_type == "sim":
                    target_perturbation = score_input - self.value_input_function(sample)
                    estimated_score = scipy.spatial.distance.cosine(predicted_perturbation, target_perturbation)
                else:
                    estimated_score = numpy.lingalg.norm(
                        self.value_input_function(sample) + predicted_perturbation, score_input
                    )

                if self.multi_feature_input:
                    return_values.append(
                        [
                            [transformer_index, input_index],
                            transformer,
                            action,
                            None,
                            None,
                            estimated_score,
                        ]
                    )
                else:
                    return_values.append([None, transformer, action, None, None, estimated_score])

        return return_values
