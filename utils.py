import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 512',type=int,default = 512)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 10',type=int,default = 10)

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of neural networks, default = 128',type=int,default = 128)

	parser.add_argument('--model', action='store', dest='model',help='model type {0 , 1 , 2}, default = 0',type=int, default = 0)

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = 0',default = '0')

	return parser
