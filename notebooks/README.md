# Demo Notebooks

We've included some demo notebooks for generating HMDA adversarial examples. Four of the notebooks generate and save the adversarial example. The last notebook reads the saved adversarial samples and outputs some metrics. All notebooks should be runnable without access to a GPU. The Simulated annealing experiments will take the longest (about 30-50 mins per model). For Brute Force and Lookup Table, the random forest model results will probably 50-60 mins.

Note that the model checkpoints were generated with an older library. Therefore, to run these notebooks, you may need to downgrade some of the libaries in your environment. We have included a different setup.py file that can be copied to the top-level directory and installed using `pip install -e .`.