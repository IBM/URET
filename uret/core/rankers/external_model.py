from uret.core.rankers.ranking_algorithm import RankingAlgorithm


class ExternalModel(RankingAlgorithm):
    def __init__(self, transformer_list, multi_feature_input=False, is_trained=True, model=None):
        """
        :param transformer_list: List of available transformations to be used by the algorithm.
        :param multi_feature_input: A boolean flag indicating if each index in the transformer
            list defines a transformer(s) for a single data type in the input. If False, then it
            is assumed that the transformer list defines multiple transformers for a single input
        :param model: A pretrained model who takes in an input sample and outputs the estimated score for every possible action
        """
        super().__init__(transformer_list, multi_feature_input, is_trained=True)

        if model is None:
            raise ValueError("A model must be provided")
        self.model = model

        # Enumerate all possible actions
        self.action_list = []
        for transformer_index, [transformer, input_index] in enumerate(transformer_list):
            if self.multi_feature_input:
                self.action_list.append(
                    [[transformer_index, input_index], transformer, action] for action in transformer.get_action_list()
                )
            else:
                self.action_list.append([None, transformer, action] for action in transformer.get_action_list())

    def rank_edges(self, sample, scoring_function, score_input, dependencies=[], current_transformation_records=None):
        scores = self.model(sample)[0]
        return [self.action_list[i] + [None, None] + s for i, s in enumerate(scores)]
