#  The Synthetic return learner reuses the Impala update rule
#  We use the contribution function of the SR to augment the timestep.reward
#  And feed the augmented timestep to the Impala Learner
from impala.learner import Learner
