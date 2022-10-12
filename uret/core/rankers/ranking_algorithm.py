from abc import ABC, abstractmethod
import warnings


class RankingAlgorithm(ABC):
    """
    This callable class implement an edge ranking algorithm used by the graph exploration algorithms.
    It is basically a function that takes a sample and generate neighbors; potentially similar
    and adversarial samples, and/or those samples that can be further used to explore more samples.
    A subclass should implement `neighbors`. The exploration algorithms use this as a callable,
    and `__call__` just invokes `neighbors`.

    Attributes
    ----------
    transformer_list: List of available transformations to be used by the algorithm.
    """

    def __init__(self, transformer_list, multi_feature_input=False, is_trained=True):
        """
        :param transformer_list: List of available transformations to be used by the algorithm. Each entry should be a
        transformers or [transformer_object, input_index] item
        :param multi_feature_input: A boolean flag indicating if each index in the transformer
        list defines a transformer(s) for a single data type in the input. If False, then it
        is assumed that the transformer list defines multiple transformers for a single input
        :param dependencies: A list of lists. Each element is the for [dependency_function, **dependency_args]. These functions
        enforce inter-feature constraints that much be maintained when a feature is modified.
        :param trained: A boolean value indicating if the ranking algorithm needs training. This is not necessary for every
        algorithm
        """

        if not isinstance(transformer_list, list):
            raise ValueError(f"Transformer list must be a list of transformers or [transformer, input_index] items")

        # Reformat transformer list if necessary
        if not isinstance(transformer_list[0], list):
            if multi_feature_input:
                raise ValueError(
                    f"For multi-feature inputs, the transformer list must be a list of [transformer, input_index] items"
                )
            self.transformer_list = []
            for transformer in transformer_list:
                self.transformer_list.append([transformer, None])
        else:
            self.transformer_list = transformer_list
        self.multi_feature_input = multi_feature_input

        self.is_trained = is_trained

    @abstractmethod
    def rank_edges(self, sample, scoring_function, score_input, dependencies=[], current_transformation_records=None):
        raise NotImplementedError

    def _train(self, training_samples, value_function, value_input_function, scoring_alg, dependencies=[], **kwargs):
        """
        Train the ranking algorithm. This is used by GraphExplorer since it holds some of the values we need for training
        :param training_samples: Input samples used to train the action dictionary
        :param value_function: Function used to compute values to compute the value a transformation has on the input with respect to a certain loss.
        :param value_input_function: Function used to compute the initial state of the sample before transform. This is fed into the value function as an input.
        :param scoring_alg: Indicates the scoring alg used for ranking if necessary
        :param dependencies: Input dependencies that need to be enforced
        """
        print(f"Training method not defined")

    def _enforce_dependencies(self, sample, dependencies):
        """
        Enforces dependencies specified in the list. Each element in dependencies should be of the form
        [callable_dependency function, dependency arguments]
        """

        if not self.multi_feature_input and len(dependencies) != 0:
            warnings.warn(
                f"Usually, dependencies are provided for multi-feature inputs. We'll keep going, but watch out!"
            )

        for d in dependencies:
            sample = d[0](sample, **d[1])
        return sample
