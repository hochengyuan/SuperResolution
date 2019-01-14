from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import pickle

from FSRCNN.model import FSRCNN

class FSRCNN_Run(object):
	def __init__(self , argument , train_loader , test_loader):
		super(FSRCNN_Run , self).__init__()
		self.upscale_factor = argument.upscale_factor
		self.lr = argument.lr
		self.epochs = argument.epochs
		self.model = None
		self.seed = argument.seed
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = None
		self.momentum = argument.momentum
		self.log_interval = argument.log_interval
		self.upscale_factor = argument.upscale_factor
		self.criterion = None
		# GPU
		self.cuda = argument.cuda
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# Result
		self.psnrList = []

	def train(self , epoch):
		self.model.train()
		epoch_loss = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = Variable(data)
			target = Variable(target)
			if self.cuda:
				data = data.cuda()
				target = target.cuda()
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, target)
			epoch_loss += loss.item()
			loss.backward()
			self.optimizer.step()
			#ToDo: Print out number of patches trained
			if batch_idx % self.log_interval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
					format(epoch, batch_idx * len(data), len(self.train_loader.dataset),
						100. * batch_idx / len(self.train_loader), loss.item()))

	def test(self):
		self.model.eval()
		sum_psnr = 0
		for data, target in self.test_loader:
			data, target = Variable(data), Variable(target)
			if self.cuda:
				data = data.cuda()
				target = target.cuda()
			output = self.model(data)
			mse = self.criterion(output , target)
			psnr = 10 * math.log10(1 / mse.item())
			sum_psnr += psnr

		self.psnrList.append(sum_psnr/len(self.test_loader))
		print("Avg. PSNR: {:.6f} dB".format(sum_psnr / len(self.test_loader)))


	def run(self):
		self.model = FSRCNN(num_channels=1 , d=56 , s=16 , m=3 , upscale_factor=self.upscale_factor).to(self.device)
		# Set convolution layer with learning rate of 10^-3
		conv1_param = list(map(id , self.model.conv1.parameters()))
		conv2_param = list(map(id , self.model.conv2.parameters()))
		conv3_param = list(map(id , self.model.conv3.parameters()))
		conv4_param = list(map(id , self.model.conv4.parameters()))
		base_param = filter(lambda p: id(p) not in conv1_param + conv2_param + conv3_param + conv4_param , self.model.parameters())

		self.model.weight_initialization(0.0 , 0.01)
		
		self.optimizer = optim.SGD([
			{'params' : base_param} , 
			{'params' : self.model.conv1.parameters() , 'lr' : self.lr*10},
			{'params' : self.model.conv2.parameters() , 'lr' : self.lr*10},
			{'params' : self.model.conv3.parameters() , 'lr' : self.lr*10},
			{'params' : self.model.conv4.parameters() , 'lr' : self.lr*10},
			],
			lr=self.lr , momentum=self.momentum)
		self.criterion = nn.MSELoss()
		if self.cuda:
			self.criterion = criterion.cuda()
		torch.manual_seed(self.seed)
		if self.cuda:
			torch.cuda.manual_seed(self.seed)

		epochList = []
		for epoch in range(1, self.epochs + 1):
			epochList.append(epoch)
			self.train(epoch)
			self.test()
			FSRCNN_Path = 'FSRCNN/batch50_56_16_3/'
			# Save the model
			model_file = FSRCNN_Path+'model_' + str(epoch) + '.pth'
			torch.save(self.model.state_dict(), model_file)
		
		# Find out the model with the highest average PSNR
		maxPSNR = max(self.psnrList)
		bestModel_index = self.psnrList.index(maxPSNR)
		print('Best Model :' , FSRCNN_Path+'model_'+str(bestModel_index + 1)+'.pth')
		# Save epochList and psnrList
		with open(FSRCNN_Path+'epochList.pkl' , 'wb') as f:
			pickle.dump(epochList , f)
		with open(FSRCNN_Path+'psnrList.pkl' , 'wb') as f:
			pickle.dump(self.psnrList , f)
		print('Saved the epochList.pkl and psnrList.pkl in' , FSRCNN_Path)