from uret.core.explorers.graph_explorer import GraphExplorer
from uret.core.rankers import Random
import simanneal
import warnings

import copy
import random
import numpy as np


class SimulatedAnnealingSearchGraphExplorer(GraphExplorer, simanneal.Annealer):
    """
    This class implement simulated annealing (random walk based on an energy function) using `simanneal` package.
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
        attack_time=0.5,
        restarts=0,
        schedule=None,
        schedule_steps=100,
        min_transform_c_sampled=1,
        min_transform_i_sampled=1,
        global_max_transforms=None,
        max_transform_c_sampled=None,
        max_transform_i_sampled=None,
        repeat_probability = 0.5,
        verbose=False,
    ):
        """
        Create a graph explorer that uses the beam search algorithm to generate adversarial samples

        :param model_predict: Classification model that the adversarial sample is being generated for. This should just be the callable prediction function of the model. It is assume it outputs a 1-hot vector or a class probability vector
        :param ranking_algorithm: Algorithm used to estimate the value of the neighboring vertices
        :param feature_extractor: Algorithm that converts a raw input into a input that the model can ingest
        :param scoring_alg: What type of loss will be used to score a vertex. This informs the explorer what parameters should be passed to the scoring function
        :param scoring_function: Function used to score a vertex (i.e. input sample). This should take in 2 or 3 arguments depending on the loss type and return a value in which, the lower the value, the "better" the index at achieving the goal
        :param dependencies: A list of functions in the form [callable, callable_args] that enforce input dependencies (e.g. features 1 and 2 should always add up to feature 3)
        :param target_label: Set this value to a class index in the model's output range to perform a targeted attack. If none, it is assume that we are performing an untargeted attack by default.
        :param attack_time: Time to spend to perform annealing. Note that if auto scheduling is used, the initial schedule search will be longer than the attack time.
        :param restarts: Number of times to perform annealing
        :param schedule: Annealing specific param. Can be provided by the user if auto schedule is not desired
        :param schedule_steps: If autoscheduling is used, this is the number of steps.
        :param min_transform_c_sampled: Min number of transformers that use the current state to randomly select
        :param min_transform_i_sampled: Min number of transformers that use the initial state to randomly select
        :param global_max_transforms: Max number of transforms to apply regardless of state. For multi-feature inputs, this will trigger a state reset if the randomly selected transforms will modifying more features than global_max_transforms. For single inputs, this will trigger a reset after move() has been called global_max_transforms times.
        :param max_transform_c_sampled: Max number of transformers that use the current state to randomly select
        :param max_transform_i_sampled: Max number of transformers that use the initial state to randomly select
        :param repeat_probability: If a transformation is to be repeated and it doesn't have a max_action constraint, flip a 
        weighted coin until failures to chose a number of repeats.
        :param verbose: disable simanneal output if false
        """

        if not isinstance(ranking_algorithm, Random):
            raise ValueError(f"The ranking algorithm must be Random")

        super().__init__(
            model_predict=model_predict,
            ranking_algorithm=ranking_algorithm,
            feature_extractor=feature_extractor,
            scoring_alg=scoring_alg,
            scoring_function=scoring_function,
            dependencies=dependencies,
            target_label=target_label,
        )

        # Get the inds for tranformer based on whether they transform the intial or current state
        # Also track which transformers must be called multiple times (e.g. string transformers)
        self.transform_initial_inds = []
        self.transform_current_inds = []
        self.transform_repeat_inds = []
        for transformer_index, (transformer, _) in enumerate(self.ranking_algorithm.transformer_list):
            if transformer.transform_initial:
                self.transform_initial_inds.append(transformer_index)
            else:
                self.transform_current_inds.append(transformer_index)

            if transformer.transform_repeat:
                self.transform_repeat_inds.append(transformer_index)

        # Simanneal params
        self.attack_time = attack_time
        self.restarts = restarts
        self.schedule = schedule  # This is automatically tuned
        self.schedule_steps = schedule_steps  # This is automatically tuned

        # Sampling Params
        self.global_max_transforms = global_max_transforms
        self.move_calls = 0
        self.min_transform_i_sampled = min(min_transform_i_sampled, len(self.transform_initial_inds))
        self.max_transform_i_sampled = (
            max_transform_i_sampled
            if max_transform_i_sampled and max_transform_i_sampled <= len(self.transform_initial_inds)
            else len(self.transform_initial_inds)
        )

        self.min_transform_c_sampled = min(min_transform_c_sampled, len(self.transform_current_inds))
        self.max_transform_c_sampled = (
            max_transform_c_sampled
            if max_transform_c_sampled and max_transform_c_sampled <= len(self.transform_current_inds)
            else len(self.transform_current_inds)
        )

        # Catch bad values of max
        if (
            self.max_transform_i_sampled < self.min_transform_i_sampled
            or self.max_transform_c_sampled < self.min_transform_c_sampled
        ):
            warnings.warn(
                f"A max transform value was less than the min transforms. Automatically setting it equal to min transforms. Be careful next time!"
            )
            self.max_transform_i_sampled = max(self.max_transform_i_sampled, self.min_transform_i_sampled)
            self.max_transform_c_sampled = max(self.max_transform_c_sampled, self.min_transform_c_sampled)

        self.repeat_prob = repeat_probability
        self.verbose = verbose

    def move(self):
        """
        Auxiliary function used by `anneal` function. This function randomly changes `self.state` using the transformations.
        """
        transform_current_inds = []
        transform_initial_inds = []

        # Get the transformers to use
        if len(self.transform_current_inds) > 0:
            transform_current_inds = random.sample(
                self.transform_current_inds,
                k=random.randint(self.min_transform_c_sampled, self.max_transform_c_sampled),
            )

        if len(self.transform_initial_inds) > 0:
            transform_initial_inds = random.sample(
                self.transform_initial_inds,
                k=random.randint(self.min_transform_i_sampled, self.max_transform_i_sampled),
            )

        all_inds = transform_current_inds + transform_initial_inds

        
        if self.global_max_transforms:
            all_inds = random.sample(all_inds, k=min(len(all_inds), self.global_max_transforms))
            # Reset the state for multi feature inputs if the new transformations would go past the global max
            if self.ranking_algorithm.multi_feature_input:
                prev_modified_inds = [i for i, tr in enumerate(self.state["transformation_records"]) if tr is not None]
                to_be_modified_inds = set(all_inds).union(prev_modified_inds)
                if len(to_be_modified_inds) > self.global_max_transforms:
                    self.state = {"sample": copy.deepcopy(self.initial_sample_state), "transformation_records": copy.deepcopy(self.initial_record_state)}
            # Reset the state for single inputs if the new transformations would go past the global max
            elif not self.ranking_algorithm.multi_feature_input:
                if self.move_calls >= self.global_max_transforms:
                    self.state = {"sample": copy.deepcopy(self.initial_sample_state), "transformation_records": copy.deepcopy(self.initial_record_state)}
                    self.move_calls = 0
                self.move_calls += 1
                    
                

        # Transform
        for i in all_inds:
            # Check if the transformer must be repeated. If so, repeat some number of times.
            if i in self.transform_repeat_inds:
                if "max_actions" in self.ranking_algorithm.transformer_list[i][0].input_constraints.keys():
                    max_actions = self.ranking_algorithm.transformer_list[i][0].input_constraints["max_actions"]
                    repeat_transform = random.randint(1, max_actions)
                # Otherwise, just repeat some number of times by flipping a coin
                else:
                    repeat_transform = 1
                    while random.random() < self.repeat_prob:
                        repeat_transform += 1

            else:
                repeat_transform = 1

            # Get the intial transform state
            if i in self.transform_initial_inds:
                transformed_state = copy.deepcopy(self.initial_sample_state)
                transformation_record = None
            else:
                transformed_state = copy.deepcopy(self.state["sample"])
                transformation_record = copy.deepcopy(self.state["transformation_records"])

            # Transform
            while repeat_transform > 0:
                return_values = self.ranking_algorithm.rank_edges(
                    transformed_state,
                    self.scoring_function,
                    self.score_input,
                    dependencies=self.dependencies,
                    current_transformation_records=transformation_record,
                    transformer_index=i,
                )
                
                if len(return_values) == 0:
                    return

                indicies, _, _, transformed_state, transformation_record, _ = return_values[
                    0
                ]  # We only expect 1 return value
                repeat_transform -= 1

            if self.ranking_algorithm.multi_feature_input:
                self.state["sample"][indicies[1]] = transformed_state[indicies[1]]
                self.state["transformation_records"][indicies[0]] = transformation_record[indicies[0]]
            else:
                self.state["sample"] = transformed_state
                self.state["transformation_records"] = transformation_record

    def search(self, sample, score_input):
        """
        Perform simulated annealing and return the lowest energy sample
        """

        convert_back_to_list = False
        if self.ranking_algorithm.multi_feature_input and isinstance(sample, list):
            convert_back_to_list = True
            sample = np.array(sample)

        # Create transformation record
        init_transformation_records = None
        if self.ranking_algorithm.multi_feature_input:
            init_transformation_records = [None for _ in range(len(self.ranking_algorithm.transformer_list))]

        self.initial_sample_state = copy.deepcopy(sample)
        self.initial_record_state = copy.deepcopy(init_transformation_records)
        self.state = {"sample": copy.deepcopy(self.initial_sample_state), "transformation_records": copy.deepcopy(self.initial_record_state)}
        self.score_input = score_input

        if self.schedule is None:
            self.schedule = self.auto(minutes=self.attack_time, steps=self.schedule_steps)
            self.set_schedule(self.schedule)

        for i in range(self.restarts + 1):
            self.initial_sample_state = copy.deepcopy(sample)
            self.initial_record_state = copy.deepcopy(init_transformation_records)
            self.state = {"sample": copy.deepcopy(self.initial_sample_state), "transformation_records": copy.deepcopy(self.initial_record_state)}
            state, energy = self.anneal()
            if convert_back_to_list:
                yield list(state["sample"]), state["transformation_records"], energy
            else:
                yield state["sample"], state["transformation_records"], energy

    def energy(self):
        """
        returns the energy to be minimized. This function is used by `simanneal` package in functions such as `self.auto` and `self.anneal`.
        """
        return self.scoring_function(self.state["sample"], self.score_input)

    def update(self, *args, **kwargs):
        """
        Wrapper for internal update. Prints out current annealing values
        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        if self.verbose:
            self.default_update(*args, **kwargs)
