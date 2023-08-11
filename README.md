# Universal Robustness Evaluation Toolkit (for Evasion) - URET

This repository contains the code for adversarial example generation tools described in:

**[Universal Robustness Evaluation Toolkit (for Evasion)](https://www.usenix.org/system/files/usenixsecurity23-eykholt.pdf)**

*Kevin Eykholt, Taesung Lee, Doug Schales, Jiyong Jang, Ian Molloy, Masha Zorin*

# Installation

Run `pip install -e .` in the top level directory. 

## Toolkit Description

Machine learning models are known to be vulnerable to adversarial evasion attacks as illustrated by image classification models. Thoroughly understanding such attacks is critical in order to ensure the safety and robustness of critical AI tasks. However, most adversarial attacks are difficult to deploy against a majority of AI systems because they have focused on image domain with only few constraints. 

URET is a solution that enables users to evaluate their models against adversarial evasion attacks regardless of data representation or model architecture. In order to generate adversarial examples for a chosen model and data domain, a user does the following:

1. Select/Define one or more **Data Transformers**.
2. Select one or more **Explorer Configurationss** and define its exploration parameters.
3. Load a pre-trained model, the set of samples to be adversarially transformed, and (optionally) a feature extractor.
4. Define any **data constraints** or **data interdependencies**
5. Run URET

To ease evaluation, URET exposes a configuration file interface.

### Data Transformers
A data transformer is an object containing function definitions for how a specific data type can be transformed. The current repository contains pre-defined transformers for some basic data types: numerical, categorical, and string. Data transformers can be combined to support other data representations. For example, tabular data may contain a combination of numerical and categorical data, which URET can support.

For unique data types, URET also exposes a transformation interface so users can define their own data transformers. The generation algorithms are fully compatible with the transformation interface so URET can be extended to support new representations. We include an example by re-coding some binary transformations (https://github.com/endgameinc/gym-malware) as a data transformer.

### Explorer Configurations
In the paper, we characterize our generation process as a graph exploration algorithm. An input is represented as a vertex and our goal is to find a series of edges (representing input transformations) that achieve the adversarial goal. An Explorer is defined by three components: the vertex scoring funtion, the edge ranking algorithm, and the search algorithm.

The **vertex scoring** function informs URET how to evaluate the fitness of explored vertices. URET will *minimize* this score, therefore lower scores = better fitness. URET includes two vertex scoring functions by default:
- **Classifier Loss**: The vertex score is equal to the output loss of the classifier. The default implementation uses cross-entropy loss, a common choice for adversarial evasion attacks. Note that if targeted adversarial examples are being generated, the negative cross-entropy loss is used.
- **Feature Distance**: The vertex score is equal to the distance between a target feature representation and the feature representation of the current input vertex. The default implementation uses the cosine distance. With this distance function, URET can integrate with existing adversarial evasion attacks that  typically target the input to the model (i.e. the output of a feature extraction pipeline). To do so, first run the adversarial evasion attack to obtain a set of target adversarial features. Then, provide the target adversarial features to URET and it will try to find adversarial inputs with the adversarial features as reference.

The **edge ranking algorithm** dicates how URET will estimate/evaluate the edge weight to neighboring vertices using the vertex scoring function. URET includes four default ranking algorithms:
- **Random**: Do not estimate/evaluate edge weights. All edges are ranking equally. Randomly return one or more edges.
- **Brute-Force**: For each connected edge, apply the corresponding input transformation and evaluate the vertex score of the visited vertex. Set the edge weight equal to the vertex score.
- **Lookup-Table**: This algorithm performs a pre-training phase on a small set of training samples before adversarial example generation. For each training sample, evaluate the edge weights of its 1-hop neighborhood using the Brute-Force algorithm. The compute edge weights are stored in a Lookup Table by (edge transformation, average edge weight). During adversarial example generation, the edge weight of an outgoing edge is obtained from the lookup table.
- **Model-Guided**: This algorithm relies on a pre-trained model to estimate edge weights. Given an edge, the model outputs the estimated edge weight. URET includes an example using reinforcement learning.

Given the edge rankings/edges returned by the edge ranking algorith, the **search algorithm** determines which neighboring vertices to further explore. URET includes two search algorithms by default:
- **Beam Search**: This algorithm keeps the best ranking edges each exploration epoch. The beam width dicates how many edges are kept and the beam depth dictates the maximum sequence length.
- **Simulated Annealing**: This is a temperature-guided time restricted random search algorithm. A user can set a time budget, which controls the per sample exploration time budget during adverarial example generation. Note that this algorithm performs a hyperparameter search before generation, which may run longer than the set time budget. During each exploration epoch, the algorithm selects a random length of random edges, applies the corresponding input transformations to the current input state, evaluates the fitness of the transformed sample, and determines if the sample should be kept (i.e. set as the current state) or discarded based on the temperature parameter. The temperature parameters balances exploration vs exploitation and decreases over time. 

Note that the simulated annealing search algorithm is only compatible with the Random edge ranking algorithm.

# Notebook Examples

In `notebooks/`, we have included several notebooks to reproduce most of the HMDA results in the paper. We have also included the configuration files, model checkpoints, and HMDA samples used for adversarial example generation. Finally, there are pre-computed adversarial samples generated from each of the notebooks.

**HMDA_random.yml** - This runs the naive random algorithm with a beam width of 5 and depth of 2.
**HMDA_brute.yml** - This runs the Brute force algorithm with a beam width of 5 and depth of 2.
**HMDA_lookup.yml** - This runs the Lookup table algorithm with a beam width of 5 and depth of 2.
**HMDA_simanneal.yml** - This runs the simanneal algorithm with a max transform limit of 2.

**HMDA_results.yml** - This reports the sucess rate and adversage transformation rate of the generated samples.

## Notebook Setup

Note that the model checkpoints were generated with an older library. Therefore, to run these notebooks, you may need to downgrade some of the libaries in your environment. Also, make sure to use Python 3.8 or greater.

1. `cp notebooks/setup.py .` - Copies the setup script for the notebooks to the top-level directory
2. `pip install -e .` - Reinstalls URET with the downgraded libraries.

After these steps, the notebooks should be runnable from the `notebooks/` directory.

# Future Development

The toolkit is under continuous development. URET's default tooling is intended to support a wide range of common machine learning scenarios, but we plan to expand the tooling based on new user needs and state-of-the-art research. Feedback, bug reports and contributions are very welcome!

# Acknowledgement
This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. Distribution Statement "A" (Approved for Public Release, Distribution Unlimited)
