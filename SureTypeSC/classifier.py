import cPickle
from .MachineLearning import Trainer, SerializedTrainer, starting_procedure_GM12878,tm_routine,evaluate_metrics
import pandas as pd
import DataLoader
import MachineLearning 




def loader(filename):
	with open (filename,'rb') as input_file:
                print input_file
		classif = SerializedTrainer(cPickle.load(input_file))
	return classif
