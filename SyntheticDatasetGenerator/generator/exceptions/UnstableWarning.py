

class UnstableWarning(Warning):
	"""The warning states that the generated process is unstable.
	"""
	
	def __init__(self, message = "The generated process is unstable."):
		self.message = message
		super().__init__(self.message)