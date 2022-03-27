# Python imports

# External imports

# Project imports

class HyperparameterSearch(object):
	"""HyperparameterSearch"""
	
	def __init__(self, resume: bool = False,
				 ):
		super().__init__()
	
	def _runSkoptOptimization(self, search_folder: str,
							  filename: str):
		checkpoint_saver = CheckpointSaver("searches/" + search_folder + "/" + filename + ".pkl", compress=9)
		
		if HAS_TO_LOAD_CHECKPOINT:
			previous_checkpoint = skopt.load(
				"searches/" + search_folder + "/" + filename + ".pkl")
			x0 = previous_checkpoint.x_iters
			y0 = previous_checkpoint.func_vals
			results = skopt.gp_minimize(objective,
										SEARCH_SPACE,
										x0=x0,
										y0=y0,
										n_calls=CALLS,
										n_initial_points=INITIAL_STARTS,
										callback=[checkpoint_saver])
		else:
			results = skopt.gp_minimize(objective,
										SEARCH_SPACE,
										n_calls=CALLS,
										n_initial_points=INITIAL_STARTS,
										callback=[checkpoint_saver])
		
		return results
