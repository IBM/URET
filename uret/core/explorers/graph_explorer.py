from abc import ABC, abstractmethod

import numpy as np
import random
import tqdm


def create_default_loss_func(scoring_alg, feature_extractor, model_predict, target_label=None):
    """
    Creates a scoring function based on the loss type
    """

    if scoring_alg == "model_loss":

        def loss_function(x, original_pred):
            """
            Default loss function that returns the probability of the current input for the correct
            label or for the target label if provided.
            param original_pred: Prediction on the original input before any modification
            """
            model_prediction = model_predict(feature_extractor(x))
            if len(np.shape(model_prediction)) == 2:
                model_prediction = model_prediction[0]

            if len(model_prediction) != 1:
                small_noise = random.uniform(0.00001, 0.000001)  # Avoid divide by zero or all/nothing classifiers
                if target_label is None:
                    return np.log(model_prediction[np.argmax(original_pred)] + small_noise)
                else:
                    return -np.log(model_prediction[target_label] + small_noise)
            elif not isinstance(model_prediction, float):
                small_noise = random.uniform(0.001, 0.0001)  # Compensate for lack of prob
                if target_label is None:
                    if model_prediction[0] == original_pred[0]:
                        print("a")
                        return np.log(1 - small_noise)
                    else:
                        print("b")
                        return np.log(small_noise)
                else:
                    if model_prediction[0] == target_label:
                        return np.log(small_noise)
                    else:
                        return np.log(1 - small_noise)
            else:
                raise ValueError(
                    f"Model predict function appears to return a single float value. Please provide a custom model loss function as there is no default implementation for this case"
                )

    else:
        import scipy.spatial

        def loss_function(x, target_feature):
            """
            Default loss function that returns the similarity of the current input's features with respect
            to the target features
            param target_feature: Target features to attain
            """
            return 1-scipy.spatial.distance.cosine(feature_extractor(x), target_feature)

    return loss_function


class GraphExplorer(ABC):
    """
    Abstract base class for a graph explorer. This object explores the transformation graph and returns possibly adversarial samples
    """

    def __init__(
        self,
        model_predict,
        ranking_algorithm,
        feature_extractor=lambda x: x,
        scoring_alg="model_loss",
        scoring_function=None,
        dependencies=None,
        target_label=None,
    ):
        """
        Create a GraphExplorer

        param model_predict: Classification model that the adversarial sample is being generated for. This should just be the callable prediction function of the model. It is assume it outputs a 1-hot vector or a class probability vector
        param ranking_algorithm: Algorithm used to estimate the value of the neighboring vertices
        param feature_extractor: Algorithm that converts a raw input into a input that the model can ingest
        param scoring_alg: What type of loss will be used to score a vertex. This informs the explorer what parameters should be passed to the scoring function
        param scoring_function: Function used to score a vertex (i.e. input sample). This should take in 2 arguments and return a value in which, the lower the value, the "better" the index at achieving the goal
        param dependencies: A list of functions in the form [callable, callable_args] that enforce input dependencies (e.g. features 1 and 2 should always add up to feature 3)
        param target_label: Set this value to a class index in the model's output range to perform a targeted attack. If none, it is assume that we are performing an untargeted attack by default.
        """

        self.model_predict = model_predict
        self.ranking_algorithm = ranking_algorithm

        self.feature_extractor = feature_extractor

        self.scoring_alg = scoring_alg.lower()
        self.scoring_function = scoring_function
        self.target_label = target_label

        if not self.scoring_alg == "model_loss" and not self.scoring_alg == "feature_loss":
            raise ValueError(self.scoring_alg + "is an unsupported loss type")

        if scoring_function is None:
            if (
                self.scoring_alg == "model_loss"
                and self.target_label is not None
                and not isinstance(self.target_label, int)
            ):
                raise ValueError("The target label must be a single int if provided.")
            scoring_function = create_default_loss_func(
                self.scoring_alg, self.feature_extractor, self.model_predict, self.target_label
            )
        self.scoring_function = scoring_function

        if dependencies is None:
            dependencies = []
        if len(dependencies) != 0 and not isinstance(dependencies[0], list):
            raise ValueError(f"Dependencies is expected to be a list of lists")
        self.dependencies = dependencies

    @abstractmethod
    def search(self, sample, score_input, *args, **kwargs):
        """
        Run the search algorithm for the sample and return the transformed sample.
        sample: The sample to transform
        score_input: An input to the scoring function used to rank an edge

        return: transformed_sample, transformation_record, estimated score of transformed sample
        """
        raise NotImplementedError

    def explore(self, x, target_features=None, return_record=False):
        """
        Generate adversarial sample(s) for x using the algorithms created during init.
        param target_features: If using feature loss, then this value contains the target features.
        param return_record: A boolean that if True will return the transformation record as well.
        """

        if self.scoring_alg == "feature_loss":
            if target_features is None:
                raise ValueError("The target features must be provided in order to use feature_loss")
            elif len(target_features) != len(x):
                raise ValueError("There must be as many target features as inputs")

        generated_samples = []

        if return_record:
            records = []

        for i, sample in enumerate(tqdm.tqdm(x)):
            original_pred = self.model_predict(self.feature_extractor(sample))
            if len(np.shape(original_pred)) == 2:
                original_pred = original_pred[0]

            best_sample = None
            best_score = np.inf
            if return_record:
                best_record = None

            if self.scoring_alg == "model_loss":
                score_input = original_pred
            else:
                score_input = target_features[i]

            for sample_next, transformation_record, _ in self.search(sample, score_input):

                # Score the current sample
                score = self.scoring_function(sample_next, score_input)

                # Early exit conditions
                # If using feature loss, then we can early exit once the target features are attained
                # Maybe consider a "close-enough" condition instead?
                if self.scoring_alg == "feature_loss" and np.array_equal(
                    target_features[i], self.feature_extractor(sample_next)
                ):
                    best_sample = sample_next
                    best_score = score
                    break

                # For all loss types, we can early exit if an adversarial example is found
                new_prediction = self.model_predict(self.feature_extractor(sample_next))
                if len(np.shape(new_prediction)) == 2:
                    new_prediction = new_prediction

                if self.target_label is not None and np.argmax(new_prediction) == self.target_label:
                    best_sample = sample_next
                    best_score = score
                    if return_record:
                        best_record = transformation_record
                    break
                elif np.argmax(new_prediction) != np.argmax(original_pred):
                    best_sample = sample_next
                    best_score = score
                    if return_record:
                        best_record = transformation_record
                    break

                # Check if the current sample is better
                if best_sample is None or score < best_score:
                    best_sample = sample_next
                    best_score = score
                    if return_record:
                        best_record = transformation_record

            if return_record:
                records.append(best_record)

            generated_samples.append(best_sample)

        if return_record:
            return generated_samples, records

        return generated_samples

    def _enforce_dependencies(self, sample):
        """
        Enforces dependencies specified in the list. Each element in dependencies should be of the form
        [callable_dependency function, dependency arguments]
        """
        for d in self.dependencies:
            sample = d[0](sample, **d[1])
        return sample

    def train(self, training_samples, value_function=None, value_input_function=None, **kwargs):
        """
        Call the training algorithm of the ranking algorithm
        """

        ## Create value functions if undefined
        if value_function is None and value_input_function is None:
            if self.scoring_alg == "model_loss":

                def value_function(x, original_pred):
                    """
                    Default value function that returns the change in probability for a given label.
                    :param original_pred: Prediction on the original input before any modification
                    """

                    model_prediction = self.model_predict(self.feature_extractor(x))
                    if len(np.shape(model_prediction)) == 2:
                        model_prediction = model_prediction[0]

                    if self.target_label is None:
                        target_label = np.argmax(original_pred)
                        return model_prediction[target_label] - original_pred[target_label]
                    # We swap the order for target label as we want lower values to reflect "good" changes when ranking edges based on how scores are processed. In the targeted label case, a "good" change is when the target label's prob increases
                    else:
                        target_label = self.target_label
                        return original_pred[target_label] - model_prediction[target_label]

            else:

                def value_function(x, original_feature):
                    """
                     Default value function that returns the change in feature for a given label
                    :param original_pred: Prediction on the original input before any modification
                    """
                    return self.feature_extractor(x) - original_feature

            def value_input_function(x):
                """
                Computes the value representation for an input
                """
                if self.scoring_alg == "model_loss":
                    model_prediction = self.model_predict(self.feature_extractor(x))
                    if len(np.shape(model_prediction)) == 2:
                        model_prediction = model_prediction[0]
                    return model_prediction
                else:
                    return self.feature_extractor(x)

        elif value_function is None or value_input_function is None:
            raise ValueError("Value_function and value_input_function must both be defined")

        self.ranking_algorithm._train(
            training_samples,
            value_function,
            value_input_function,
            self.scoring_alg,
            dependencies=self.dependencies,
            **kwargs,
        )
