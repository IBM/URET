import numpy as np

from uret.core.rankers import BruteForce, LookupTable, ExternalModel, Random
from uret.core.explorers import (
    BeamSearchGraphExplorer,
    GreedySearchGraphExplorer,
    SimulatedAnnealingSearchGraphExplorer,
)


def process_config_file(config_filepath, model, feature_extractor=None, input_processor_list=[]):
    """
    Processes a config file and returns the explorer
    config_filepath: Path to the config file to parse
    model: Model to be attacked with a function that returns the prediction vector
    feature_extractor: Converts input data into a form the model can ingest
    input_processor_dict: Mapping of function names to functions
    """
    import os.path
    import yaml
    import importlib

    with open(config_filepath, "r") as stream:
        config_params = yaml.safe_load(stream)

    transformer_params = config_params.get("transformer_params", None)
    ranker_params = config_params.get("ranker_params", None)
    explorer_params = config_params.get("explorer_params", None)
    dependency_params = config_params.get("dependency_params", None)

    if transformer_params is None:
        raise RuntimeError(f"transformer_params was not found in the config file")
    if ranker_params is None:
        raise RuntimeError(f"ranker_params was not found in the config file")
    if explorer_params is None:
        raise RuntimeError(f"explorer_params was not found in the config file")

    # 1.Create the Transformer List
    # Assume transformers for data types are in the uret.transformers directory
    transformer_list = []
    for tp in transformer_params:
        data_type = tp["data_type"]
        init_args = tp.get("init_args", {})

        if data_type.lower() == "category" or data_type.lower() == "number" or data_type.lower() == "string":
            transformer_path = "uret.transformers.basic"
        else:
            transformer_path = "uret.transformers." + data_type.lower()

        transformer_module = importlib.import_module(transformer_path)
        transformer = getattr(transformer_module, data_type.lower().capitalize() + "Transformer")
        input_processor_name = tp.get("input_processor_name", None)
        if input_processor_name is not None:
            found = False
            for ip in input_processor_list:
                if ip.__name__ == input_processor_name:
                    init_args["input_processor"] = ip
                    found = True
                    break
            if not found:
                raise ValueError(f"No entry for {input_processor_name} in input_processor_dict")

        if len(init_args.keys()) > 0:
            transformer = transformer(**init_args)
        else:
            transformer = transformer()

        feature_index = tp.get("feature_index", None)
        if (
            "init_args" in ranker_params
            and ranker_params["init_args"].get("multi_feature_input", False)
            and feature_index is None
        ):
            raise ValueError(
                f"Ranker indicates multi-feature input transformations, but the transformer doesn't specify which feature index it modifies"
            )

        transformer_list.append([transformer, feature_index])

    # 2.Create the Ranker
    if ranker_params["type"].replace("_", "").lower() in ["brute", "bruteforce"]:
        ranking_alg = BruteForce
    elif ranker_params["type"].replace("_", "").lower() in ["lookup", "lookuptable"]:
        ranking_alg = LookupTable
    elif ranker_params["type"].replace("_", "").lower() in ["external", "externalmodel"]:
        ranking_alg = ExternalModel
    elif ranker_params["type"].lower() == "random":
        ranking_alg = Random
    else:
        raise ValueError(f"{ranker_params['type']} is not a recognized ranker type")

    if "init_args" in ranker_params.keys():
        ranker = ranking_alg(transformer_list, **ranker_params["init_args"])
    else:
        ranker = transformer(transformer_list)

    # 3.Create the Explorer and return
    explorer_init_args = explorer_params.get("init_args", {})
    explorer_init_args["ranking_algorithm"] = ranker

    # Create model values
    predict_function_name = explorer_params.get("predict_function_name", "predict")
    if not hasattr(model, predict_function_name):
        raise ValueError(f"{predict_function_name} is not an attribute of model")
    explorer_init_args["model_predict"] = getattr(model, predict_function_name)
    explorer_init_args["feature_extractor"] = feature_extractor

    # Create dependencies
    if dependency_params is not None:
        dependencies = dependency_params.get("dependencies", [])
        if len(dependencies) > 0:
            dependency_path = (
                "utils.dependency_functions." + dependency_params.get("dependency_path", "default").split(".")[0]
            )
            dependency_module = importlib.import_module(dependency_path)
            explorer_init_args["dependencies"] = []
            for d in dependencies:
                d_func = getattr(dependency_module, d["name"])
                explorer_init_args["dependencies"].append([d_func, d["args"]])

    if explorer_params["type"].replace("_", "").lower() in ["beam", "beamsearch", "beamsearchgraphexplorer"]:
        explorer_alg = BeamSearchGraphExplorer
    elif explorer_params["type"].replace("_", "").lower() in ["greedy", "greedysearch", "greedysearchgraphexplorer"]:
        explorer_alg = GreedySearchGraphExplorer
    elif explorer_params["type"].replace("_", "").lower() in [
        "simanneal",
        "simulatedannealing",
        "simulatedannealingsearchgraphexplorer",
    ]:
        explorer_alg = SimulatedAnnealingSearchGraphExplorer
    else:
        raise ValueError(f"{explorer_params['type']} is not a recognized explorer type")

    return explorer_alg(**explorer_init_args)
