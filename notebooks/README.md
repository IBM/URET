# Demo Notebooks

We've included some demo notebooks for generating HMDA adversarial examples. Four of the notebooks generate and save the adversarial example. The last notebook reads the saved adversarial samples and outputs some metrics. All notebooks should be runnable without access to a GPU. The Simulated annealing experiments will take the longest (about 30-50 mins per model). For Brute Force and Lookup Table, the random forest model results will probably 50-60 mins.

**HMDA_random.yml** - This runs the naive random algorithm with a beam width of 5 and depth of 2.
**HMDA_brute.yml** - This runs the Brute force algorithm with a beam width of 5 and depth of 2.
**HMDA_lookup.yml** - This runs the Lookup table algorithm with a beam width of 5 and depth of 2.
**HMDA_simanneal.yml** - This runs the simanneal algorithm with a max transform limit of 2.

**HMDA_results.yml** - This reports the sucess rate and adversage transformation rate of the generated samples.

Note that the model checkpoints were generated with an older library. Therefore, to run these notebooks, you may need to downgrade some of the libaries in your environment. Specifically:
- Use python 3.8
- Copy the setup.py file into the top-level directory and install using `pip install -e .`.
