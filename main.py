from __future__ import print_function
import argparse
from torchvision import datasets, transforms
from data_preprocessing import load_training_data , load_testing_data
from SRCNN.train_test import SRCNN_Run
from FSRCNN.train_test import FSRCNN_Run
import timeit

parser = argparse.ArgumentParser(description='Running Super Resolution Model')
# Choose the Neural Network
parser.add_argument('--model' , type=str , default='FSRCNN' , metavar='-m' , help='key in the model name')
parser.add_argument('--upscale_factor' , type=int , default=3 , metavar='-uf' , help='key in the upscale_factor')
# Hyper Parameter
parser.add_argument('--trainData' , type=str , default='data/train' , metavar='trD' , help='file path of the training data')
parser.add_argument('--testData' , type=str , default='data/test' , metavar='teD' , help='file path of the training data')
parser.add_argument('--batch_size' , type=int , default=1 , metavar='N' , help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int , default=50 , metavar='N', help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.0001 , metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float , default=0.5 , metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed' , type=int , default=2 , metavar='S' , help='random seed (default: 2)')
parser.add_argument('--log_interval', type=int, default=10, metavar='L', help='how many batches to wait before logging training status')
# GPU or CPU
parser.add_argument('--cuda', action='store_true', help='use cuda?')

args = parser.parse_args()
if not args.cuda:
	print('There is No GPU')
else:
	print('I Have GPU')
if args.cuda and not torch.cuda.is_available():
	raise Exception("No GPU Found")


def main():
	start = timeit.default_timer()
	train_loader = load_training_data(filePath=args.trainData , batch_size=args.batch_size , upscale_factor=args.upscale_factor)
	test_loader = load_testing_data(filePath=args.testData , batch_size=args.batch_size , upscale_factor=args.upscale_factor)
	if args.model == 'SRCNN':
		model = SRCNN_Run(args , train_loader , test_loader)
	elif args.model == 'FSRCNN':
		model = FSRCNN_Run(args , train_loader , test_loader)
	else:
		raise Exception("the model does not exist!")

	model.run()
	end = timeit.default_timer()
	print('Time Consuming :' , (end-start)/60 , "minutes.")
if __name__ == '__main__':
	main()