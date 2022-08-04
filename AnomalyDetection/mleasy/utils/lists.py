

def all_indices(list_: list, arg) -> list[int]:
	"""Finds all indices of `arg` in `list`, if any.
	
	Parameters
	----------
	list_ : list
		It is a list in which we want to find occurrences of `arg`.
		
	arg : object
		It is the object we are looking for in `list_`.

	Returns
	-------
	indices : list of int
		It is the list containing all the indices of `list_` containing `arg`.
	"""
	indices = [idx
			   for idx, elem in enumerate(list_) if elem == arg]
	return indices