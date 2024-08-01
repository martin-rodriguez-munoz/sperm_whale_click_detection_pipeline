

import torch.nn as nn

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        # self.conv8 = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform.unsqueeze(1).permute(0,1,3,2))
        # print(f'conv1: {x.shape}')
        x = self.batchnorm1(x)
        # print(f'batchnorm1: {x.shape}')
        x = self.relu1(x)
        # print(f'relu1: {x.shape}')
        x = self.maxpool1(x)
        # print(f'maxpool1: {x.shape}')

        x = self.conv2(x)
        # print(f'conv2: {x.shape}')
        x = self.batchnorm2(x)
        # print(f'batchnorm2: {x.shape}')
        x = self.relu2(x)
        # print(f'relu2: {x.shape}')
        x = self.maxpool2(x)
        # print(f'maxpool2: {x.shape}')

        x = self.conv3(x)
        # print(f'conv3: {x.shape}')
        x = self.batchnorm3(x)
        # print(f'batchnorm3: {x.shape}')
        x = self.relu3(x)
        # print(f'relu3: {x.shape}')

        x = self.conv4(x)
        # print(f'conv4: {x.shape}')
        x = self.batchnorm4(x)
        # print(f'batchnorm4: {x.shape}')
        x = self.relu4(x)
        # print(f'relu4: {x.shape}')

        x = self.conv5(x)
        # print(f'conv5: {x.shape}')
        x = self.batchnorm5(x)
        # print(f'batchnorm5: {x.shape}')
        x = self.relu5(x)
        # print(f'relu5: {x.shape}')
        # x = self.maxpool5(x)

        x = self.conv6(x)
        # print(f'conv6: {x.shape}')
        x = self.batchnorm6(x)
        # print(f'batchnorm6: {x.shape}')
        x = self.relu6(x)
        # print(f'relu6: {x.shape}')

        x = self.conv7(x)
        # print(f'conv7: {x.shape}')
        x = self.batchnorm7(x)
        # print(f'batchnorm7: {x.shape}')
        x = self.relu7(x)
        # print(f'relu7: {x.shape}')

        # x = self.conv8(x)
        x = x.reshape(x.shape[0],-1)
        # print(x.shape)
        return x

class ManyLayerMlp(nn.Module):
    '''
    Get the probabilities of 0/1 clicks in the detections window
    '''
    def __init__(self, input_size):
        super(ManyLayerMlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, input_audio):
        # print(input_audio.shape)
        output = self.layers(input_audio)
        return output