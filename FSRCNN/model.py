import torch
import torch.nn as nn

class FSRCNN(torch.nn.Module):
	def __init__(self , num_channels , d , s , m , upscale_factor):
		super(FSRCNN , self).__init__()
		self.layers = []
		# Feature Extraction
		self.conv1 = nn.Conv2d(in_channels=num_channels , out_channels=d , kernel_size=5 , padding=2)
		self.prelu1 = nn.PReLU()
		# Shrinking
		self.conv2 = nn.Conv2d(in_channels=d , out_channels=s , kernel_size=1 , padding=0)
		self.prelu2 = nn.PReLU()
		self.layers.append(nn.Sequential(self.conv2 , self.prelu2))
		# Non-Linear Mapping
		self.conv3 = nn.Conv2d(in_channels=s , out_channels=s , kernel_size=3 , padding=1)
		for i in range(0,m,1):
			self.layers.append(self.conv3)
		self.prelu3 = nn.PReLU()
		self.layers.append(self.prelu3)
		# Expending
		self.conv4 = nn.Conv2d(in_channels=s , out_channels=d , kernel_size=1 , padding=0)
		self.prelu4 = nn.PReLU()
		self.layers.append(nn.Sequential(self.conv4 , self.prelu4))
		# Layering
		self.layering = nn.Sequential(*self.layers)
		# Deconvolution
		self.deconv = nn.ConvTranspose2d(in_channels=d , out_channels=num_channels , kernel_size=9 , stride=upscale_factor , padding=3)

	def forward(self, x):
		# Feature Extraction
		out1 = self.conv1(x)
		out2 = self.prelu1(out1)
		# Layering
		out3 = self.layering(out2)
		# Deconvolution
		out4 = self.deconv(out3)

		return out4

	def weight_initialization(self , mean , std):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(mean, std)
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, 0.0001)
				if m.bias is not None:
					m.bias.data.zero_()