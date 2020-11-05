from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

class DelayerScheduler(_LRScheduler):
	""" Starts with a flat lr schedule until it reaches N epochs the applies a scheduler 
	Args:
		optimizer (Optimizer): Wrapped optimizer.
		delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
		after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
	"""

	def __init__(self, optimizer, delay_epochs, after_scheduler):
     """
     Initialize the initialisation.

     Args:
         self: (todo): write your description
         optimizer: (todo): write your description
         delay_epochs: (int): write your description
         after_scheduler: (todo): write your description
     """
		self.delay_epochs = delay_epochs
		self.after_scheduler = after_scheduler
		self.finished = False
		super().__init__(optimizer)

	def get_lr(self):
     """
     Returns the current learning rate.

     Args:
         self: (todo): write your description
     """
		if self.last_epoch >= self.delay_epochs:
			if not self.finished:
				self.after_scheduler.base_lrs = self.base_lrs
				self.finished = True
			return self.after_scheduler.get_lr()

		return self.base_lrs

	def step(self, epoch=None):
     """
     Perform a single step.

     Args:
         self: (todo): write your description
         epoch: (array): write your description
     """
		if self.finished:
			if epoch is None:
				self.after_scheduler.step(None)
			else:
				self.after_scheduler.step(epoch - self.delay_epochs)
		else:
			return super(DelayerScheduler, self).step(epoch)

def DelayedCosineAnnealingLR(optimizer, delay_epochs, cosine_annealing_epochs):
    """
    Determine the learning rate of the given training training.

    Args:
        optimizer: (todo): write your description
        delay_epochs: (todo): write your description
        cosine_annealing_epochs: (bool): write your description
    """
	base_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_epochs)
	return DelayerScheduler(optimizer, delay_epochs, base_scheduler)