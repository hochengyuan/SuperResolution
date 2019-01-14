import torch.utils.data as data


from os import listdir
from os.path import join
from PIL import Image, ImageFilter


def generate_imageList(filepath):
	return [join(filepath , x) for x in listdir(filepath)]

def size_DataSet(givenList):
	return len(givenList)

class Data_Initialization(data.Dataset):
	def __init__(self , filepath , input_transform=None , target_transform=None):
		super(Data_Initialization , self).__init__()
		self.image_fileList = generate_imageList(filepath)
		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self , index):
		cur_image = Image.open(self.image_fileList[index])
		input_image = cur_image.convert('YCbCr').split()[0]
		target = input_image.copy()
		if self.input_transform != None:
			input_image = input_image.filter(ImageFilter.GaussianBlur(2))
			input_image = self.input_transform(input_image)
		if self.target_transform != None:
			target = self.target_transform(target)

		return input_image , target

	def __len__(self):
		return len(self.image_fileList)