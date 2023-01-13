import os

from mleasy.utils.json import save_py_json, load_py_json


class HyperparameterSearchSaver(object):
	"""Object saving the state of the search when called.
	
	Parameters
	----------
	file_path : str
		It is the path of the file where to save the search steps.
	"""
	__HISTORY = "_history"
	__EXT = ".checkpoint"
	
	def __init__(self, file_path: str):
		super().__init__()
		
		self.file_path = file_path
		
	def __call__(self, *args, **kwargs):
		"""Saves the information passed to file.
		
		The information are safely stored using JSON instead of using pickle,
		which has security issues and could represent a code injection problem
		if the user tries to load malicious built object.
		
		Parameters
		----------
		args
			Ignored, present for consistency with API.
		
		kwargs
			The accepted keywords are ["search_history"]. Every information is
			stored in a file with the file_path specified with appended the
			string "save_".

		Returns
		-------
		None
		"""
		if "search_history" in kwargs.keys():
			history = kwargs["search_history"]
			save_py_json(history, self.file_path + self.__HISTORY + self.__EXT)
	
	def load_history(self) -> dict:
		"""Loads the history from file and returns it.
		
		Returns
		-------
		history : dict
			The history of the search at the given path.
		"""
		history = load_py_json(self.file_path + self.__HISTORY + self.__EXT)
		
		return history if history is not None else {}
	
	def history_exists(self) -> bool:
		"""Checks if the history exists.
		
		Returns
		-------
		history_exists : bool
			True if the history exists
		"""
		if os.path.exists(self.file_path + self.__HISTORY + self.__EXT):
			return True
		else:
			return False
