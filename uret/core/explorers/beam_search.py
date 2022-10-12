from uret.core.explorers.graph_explorer import GraphExplorer
import numpy as np
import copy


class BeamSearchGraphExplorer(GraphExplorer):
    """
    This class implement Beam Search using recursion.
    """

    def __init__(
        self,
        model_predict,
        ranking_algorithm,
        feature_extractor=None,
        scoring_alg="model_loss",
        scoring_function=None,
        dependencies=None,
        target_label=None,
        search_size=3,
        max_depth=2,
        max_visits=None,
    ):
        """
        Create a graph explorer that uses the beam search algorithm to generate adversarial samples

        param model_predict: Classification model that the adversarial sample is being generated for. This should just be the callable prediction function of the model. It is assume it outputs a 1-hot vector or a class probability vector
        param ranking_algorithm: Algorithm used to estimate the value of the neighboring vertices
        param feature_extractor: Algorithm that converts a raw input into a input that the model can ingest
        param scoring_alg: What type of loss will be used to score a vertex. This informs the explorer what parameters should be passed to the scoring function
        param scoring_function: Function used to score a vertex (i.e. input sample). This should take in 2 or 3 arguments depending on the loss type and return a value in which, the lower the value, the "better" the index at achieving the goal
        param dependencies: A list of functions in the form [callable, callable_args] that enforce input dependencies (e.g. features 1 and 2 should always add up to feature 3)
        param target_label: Set this value to a class index in the model's output range to perform a targeted attack. If none, it is assume that we are performing an untargeted attack by default.
        param search_size: How many top neighbors to explore at each step/for each node.
        param max_depth: How many steps to explore, or how many times we stack `neighbor_algorithm` on a sample.
        max_visits: The max number of generated samples to visit.
        """

        super().__init__(
            model_predict=model_predict,
            ranking_algorithm=ranking_algorithm,
            feature_extractor=feature_extractor,
            scoring_alg=scoring_alg,
            scoring_function=scoring_function,
            dependencies=dependencies,
            target_label=target_label,
        )

        self.search_size = search_size
        self.max_depth = max_depth
        self.visited_nodes = []  # Keep track of visited nodes to avoid revisits
        self.max_visits = max_visits

    def search(self, sample, score_input, transformation_records=None, depth=0):
        """
         runs search and returns (sample candidate, list of applied transformations, score).
         This method implements beam sesarch in a depth-first and recursive manner, taking top-`self.search_size` samples,
         and repeating the function up to `self.max_depth` times.

         :param sample: an input sample, which can be either the raw input for the feature extractor,
                        or the feature vector produced by the feature extractor that will be fed to the model.
        :param score_input: Input value for the scoring function. This is either the original pred or the target feature value
        :param transformation_records: history of applied transformations.
        :param depth: current depth of search.
        """
        if depth >= self.max_depth:
            return
        if depth == 0:
            self.visited_nodes = []

        convert_back_to_list = False
        if self.ranking_algorithm.multi_feature_input and isinstance(sample, list):
            convert_back_to_list = True
            sample = np.array(sample)

        edge_transform_estimates = self.ranking_algorithm.rank_edges(
            sample, self.scoring_function, score_input, self.dependencies, transformation_records
        )

        # No actions are possible from current vertex
        if len(edge_transform_estimates) == 0:
            return

        indicies, transformers, transformer_params, samples_next, new_transformation_records, scores = zip(
            *edge_transform_estimates
        )

        ranked_edge_list = np.argsort(scores)  # Score is the lower, the better.
        rank = 0
        returned_edge_count = 0
        while returned_edge_count < self.search_size and rank < len(ranked_edge_list):
            if rank >= len(ranked_edge_list):
                break
            edge_idx = ranked_edge_list[rank]
            # Get the values for the edge that is being evaluated
            indicies_next = indicies[
                edge_idx
            ]  # if not none, this indicates which feature of x the transformer transforms
            transformer_next = transformers[edge_idx]
            transformer_param_next = transformer_params[edge_idx]
            sample_next = samples_next[edge_idx]
            transformation_records_next = new_transformation_records[edge_idx]

            # Apply the transformation, if not done by the neighbor eval algorithm.
            if sample_next is None:
                if indicies_next is not None:
                    sample_next = copy.copy(sample)
                    transformation_records_next = copy.copy(transformation_records)
                    if self.ranking_algorithm.multi_feature_input and transformation_records_next is None:
                        transformation_records_next = [
                            None for _ in range(len(self.ranking_algorithm.transformer_list))
                        ]

                    transformed_value, new_transformation_record = transformer_next.transform(
                        sample_next[indicies_next[1]],
                        transformation_record=transformation_records_next[indicies_next[0]],
                        transformation_value=transformer_param_next,
                    )
                    sample_next[indicies_next[1]] = transformed_value
                    transformation_records_next[indicies_next[0]] = new_transformation_record
                else:
                    sample_next, transformation_records_next = transformer_next.transform(
                        sample,
                        transformation_record=transformation_records,
                        transformation_value=transformer_param_next,
                    )

                sample_next = self._enforce_dependencies(
                    sample_next
                )  # Enforce dependencies since ranking algorithm did not

            # Only evaluate nodes that haven't been previously visited
            if not np.any(
                [np.all(sample_next == v) for v in self.visited_nodes]
            ):  # This might not work with all data types?
                if convert_back_to_list:  # Restore back to list
                    sample_next = list(sample_next)

                self.visited_nodes.append(sample_next)
                score_next = scores[edge_idx]
                yield sample_next, transformation_records_next, score_next
                returned_edge_count += 1
                if self.max_visits and len(self.visited_nodes) >= self.max_visits:
                    break
                for sts_tuple in self.search(
                    sample_next,
                    score_input,
                    transformation_records_next,
                    depth + 1,
                ):  # recursion
                    yield sts_tuple
            rank += 1


def GreedySearchGraphExplorer(
    model_predict,
    ranking_algorithm,
    feature_extractor=None,
    scoring_alg="model_loss",
    scoring_function=None,
    dependencies=[],
    target_label=None,
    max_depth=2,
):
    """
    Alias for BeamSearchGraphExplorer with search size 1, always taking the best option greedily.
    """
    return BeamSearchGraphExplorer(
        model_predict,
        ranking_algorithm,
        feature_extractor,
        scoring_alg,
        scoring_function,
        dependencies,
        target_label,
        search_size=1,
        max_depth=max_depth,
    )
