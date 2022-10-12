import numpy as np
import random

import rl
import rl.core
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.agents.dqn import DQNAgent
from rl.policy import Policy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import ModelIntervalCheckpoint

import tensorflow as tf
import keras


class RLEnv(rl.core.Env):
    def __init__(
        self,
        task,
        input_encoding,
        # fe,
        # state_x_dimension,
        # state_f_dimension=20,
        max_steps=np.inf,
    ):
        """
        @input fe: feature extractor
        @input x_train: raw training data used as input to fe
        """
        self.flattened_transformers = task.flattened_transformers

        self.task = task
        self.input_encoding = input_encoding

        # self.fe = fe
        # self.x_initial = None
        # self.x_target = None
        self.x_current = None
        # self.x_train = None
        # self.f_target = None
        self.num_steps = 0
        self.reward_prev = 0

        # self.norm_p = norm_p
        # self.state_x_dimension = state_x_dimension
        # self.state_f_dimension = state_f_dimension
        # self.observation_dimension = self.state_x_dimension + self.state_f_dimension
        self.max_steps = max_steps
        self.max_actions_applied = task.max_actions_applied
        self.num_action_kinds = task.num_action_kinds
        self.verbose = 0

        self.y_num_actions_applied = 0
        self.y_action_sequence = [0]

        self.task.__iter__()

    # def set_x_train(self, x_train):
    #     self.x_train = x_train
    #     self.x_initial = None
    #     self.x_target = None
    #     self.x_current = None
    #     self.f_target = None
    #     self.num_steps = 0
    #     self.reward_prev = 0

    def set_verbosity(self, verbose=0):
        self.verbose = verbose

    def step(self, action):
        x_prev = self.x_current
        self.x_current = self.action(action, self.x_current)
        r0 = self.reward_prev
        self.reward_prev = r1 = self.reward(self.x_current)
        reward = r1 - r0

        state = self.get_state(self.x_current)
        stop = self.is_complete(r1)
        self.num_steps += 1
        if self.verbose > 1:
            print(
                "\nStep: %s ==> %s (action=%d, reward=%f, %s)"
                % (x_prev, self.x_current, action, reward, "success" if self.x_current == self.x_target else "not yet")
            )
        if self.num_steps > self.max_steps:
            stop = True
        return state, reward, stop, {}

    def get_optimal_action(self):
        if self.num_steps < len(self.y_action_sequence):
            if self.verbose > 1:
                print(
                    "Env::get_optimal_action(policy)::num_steps=",
                    self.num_steps,
                    ", y_action_sequence=",
                    self.y_action_sequence,
                )
            return self.y_action_sequence[self.num_steps]
        else:
            if self.verbose > 1:
                print(
                    "Env::get_optimal_action(rand)::num_steps=",
                    self.num_steps,
                    ", y_action_sequence=",
                    self.y_action_sequence,
                )
            return np.random.randint(0, self.num_action_kinds)

    def get_state(self, xi):
        state = np.concatenate([self.input_encoding(xi), self.task.get_delta_target(xi)], axis=0)
        return state

    def reset(self):
        ## TODO: Training time task has different goal to the test time task.
        # In training, the target and the different are measured against x_target.
        # Can we change Task to accomodate external goals? Or is it ideal to make a RLTask?
        # self.x_target = self.x_initial = next(self.task)  # Maybe, we just need to take x_init, x_target from the task, and use these instead of Task itself. But then, can we use the score function?
        # self.y_num_actions_applied = random.randrange(1, self.max_actions_applied)
        # a_seq = []
        # for i in range(self.y_num_actions_applied):
        #     a = random.randrange(self.num_action_kinds)
        #     self.x_target = self.action(a, self.x_target)
        #     a_seq.append(a)
        # self.y_action_sequence = a_seq
        # if self.verbose > 1:
        #     print("\n  " + self.x_initial + "=(%d)=>" % self.y_num_actions_applied + self.x_target)

        self.x_current, self.y_action_sequence, self.y_num_actions_applied = next(self.task)
        # self.f_target = self.fe(self.x_target)
        self.num_steps = 0

        # Initial reward
        # f_current = self.fe(self.x_current)
        # state = np.concatenate([self.input_encoding(self.x_current), self.f_target-f_current], axis=0)
        # state = self.get_state(self.x_current, self.f_target)
        state = self.get_state(self.x_current)
        self.reward_prev = self.reward(self.x_current)
        return state

    def render(self, mode="human"):
        print("\n=>" + self.x_current + "(%f)" % self.reward_prev)

    def close(self):
        pass

    def action(self, action, x):
        """
        Applies an action specified by action on x and returns it.
        :return:
        :rtype:
        """
        t, a = self.flattened_transformers[action]
        x = t.transform(x, a)
        return x

    # def input_encoding(self, x):  # Need to be customized. Model-specific mapping to a vector.
    #     """
    #     Encodes the raw input x into a numerical form.
    #     :return:
    #     :rtype:
    #     """
    #     raise NotImplementedError

    def reward(self, x):
        """
        Returns the reward of x for the current task.
        :return:
        :rtype:
        """
        return self.task.score(x)

    def is_complete(self, r):  # Need to be customized
        """
        Returns if the reward r is its maximum score and needs to end the episode.
        :return:
        :rtype:
        """
        return r == 1


class DGAEnv(RLEnv):
    character_to_int = {}
    character_to_int.update(zip("abcdefghijklmnopqrstuvwxyz", range(26)))
    character_to_int.update(zip("0123456789", range(26, 36)))
    character_to_int["-"] = 36
    character_to_int["."] = 37

    int_to_character = {v: k for (k, v) in character_to_int.items()}

    num_inputs = 64  # 64 characters
    num_action_kinds_list = (
        1,
        12,
        len(character_to_int),
    )  # 1 action (substitution), 12 locations, 37 characters (alphabet, numbers, -)
    num_hidden = 32

    num_action_kinds = num_action_kinds_list[0] * num_action_kinds_list[1] * num_action_kinds_list[2]

    def action(self, action_encoded, x):  # Need to be customized.
        action = action_encoded // (DGAEnv.num_action_kinds_list[1] * DGAEnv.num_action_kinds_list[2])
        action_location = (
            action_encoded
            % (DGAEnv.num_action_kinds_list[1] * DGAEnv.num_action_kinds_list[2])
            // DGAEnv.num_action_kinds_list[2]
        )
        action_character = (
            action_encoded
            % (DGAEnv.num_action_kinds_list[1] * DGAEnv.num_action_kinds_list[2])
            % DGAEnv.num_action_kinds_list[2]
        )

        if action == 0:  # substitution
            dn_new = x[:action_location] + DGAEnv.int_to_character[action_character] + x[action_location + 1 :]
        return dn_new

    def input_encoding(self, x):  # Need to be customized
        state = np.zeros([self.state_x_dimension])
        state[: len(x)] = list(map(lambda c: DGAEnv.character_to_int[c], x))
        return state

    def reward(self, x, f_target):  # Need to be customized
        f = self.fe(x)
        dist = np.linalg.norm(f - f_target, self.norm_p)  # Now we have to choose the norm.
        return len(f_target) - dist  # Linear reward. len(target) is an arbitrary max value.

    def is_complete(self, r):  # Need to be customized
        return r == len(self.f_target)


# Policies


class GoldStandardPolicy(Policy):
    def __init__(self, env, is_training, on_policy):
        #         self.action_random_select = True
        self.is_training = is_training
        self.on_policy = on_policy
        self.env = env

    def select_action(self, q_values):
        if self.is_training == True and self.on_policy == True:
            action_triple = self.env.get_optimal_action()
            return action_triple
        else:
            return np.argmax(q_values)


class GSEpsGreedyQPolicy(GoldStandardPolicy):
    def __init__(self, env, is_training, on_policy, eps=0.1, gs=0.4):
        super(GSEpsGreedyQPolicy, self).__init__(env, is_training, on_policy)
        self.eps = eps
        self.gs = gs

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        p = np.random.uniform()
        if p < self.eps:
            action = np.random.randint(0, nb_actions)
        elif p >= self.eps and p < self.eps + self.gs:
            action = super(GSEpsGreedyQPolicy, self).select_action(q_values)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(GSEpsGreedyQPolicy, self).get_config()
        config["eps"] = self.eps
        config["gs"] = self.gs
        return config


def train(
    model,
    env,
    nb_steps=1750000,
    log_interval=10000,
    nb_max_episode_steps=3,
    checkpoint_name=None,
    checkpoint_interval=10000,
    linear_anneal_nb_steps=1000000,
):
    # Train
    policy = LinearAnnealedPolicy(
        GSEpsGreedyQPolicy(env, True, True, eps=0.1, gs=0.3),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=linear_anneal_nb_steps,
    )
    memory = SequentialMemory(limit=100000, window_length=1)
    if hasattr(model.output, "__len__"):  # Keras-RL & TF incompatibility workaround
        delattr(tf.Tensor, "__len__")

    dqn = DQNAgent(
        model=model,
        nb_actions=env.num_action_kinds,
        policy=policy,
        memory=memory,
        processor=None,
        nb_steps_warmup=100,
        gamma=0.99,
        target_model_update=10000,
        train_interval=1,
        delta_clip=1.0,
    )
    # dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.compile(keras.optimizers.Adam(), metrics=["mae"])

    callbacks = []
    if checkpoint_name:
        callbacks.append(ModelIntervalCheckpoint(checkpoint_name, checkpoint_interval))

    env.set_verbosity(0)
    dqn.fit(
        env,
        nb_steps=nb_steps,
        log_interval=log_interval,
        nb_max_episode_steps=nb_max_episode_steps,
        callbacks=callbacks,
    )
    return dqn
