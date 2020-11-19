import argparse
import sys
import os

class opts():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument('--mode', type = str, default = 'train', help = 'training or testing stage. Currently only support training')
		self.parser.add_argument('--batch_size', type = int, default = 8)
		self.parser.add_argument('--n_epochs', type = int, default = 5)
		self.parser.add_argument('--lr', type = float, default = 2e-4)
		self.parser.add_argument('--gpu', type = bool, default = False)
		self.parser.add_argument('--num_workers', type = int, default = 0, help = 'dataloader threads. 0 for single thread')
		self.parser.add_argument('--seed', type = int, default = 0, help = 'random seed for producibility')
		self.parser.add_argument('--resume', action = 'store_true', help = 'resume an experiment. reload optimizer and model')
		self.parser.add_argument('--load_model', type = str, default = './checkpoints/Unet_checkpoint_last.pth', help = 'path to load optmizer and model')
		self.parser.add_argument('--show_input', type = bool, default = False, help = 'while True, show input iamges(first one of every batch-size) every iter')

	def parse(self):
		opts = self.parser.parse_args()
		return opts