
class FlowError(Exception):
	"""The call triggers a flow error. Order of method calls is wrong.
	"""
	
	def __init__(self, message = "There has been a flow error."):
		self.message = message
		super().__init__(self.message)