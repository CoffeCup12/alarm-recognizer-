import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, num_categories):
        super(model, self).__init__()

        #CNN layers
        self.cnn = nn.Sequential(
            #first Conv layer
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=(2,1)),

            #second conv layer
            nn.Conv2d(output_channel, 2 * output_channel, kernel_size, padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=(2,1)),

            #flatten spatial demsion
            nn.Flatten(),
        )

        #LSTM layers
        self.lstm = nn.LSTM(128 * output_channel, 2 * output_channel, batch_first=True)

        #classification layer
        self.fc = nn.Linear(2 * output_channel, num_categories)

    def forward(self, x):

        #store input dimensions
        batch_size, channels, height, time_steps = x.size()

        #combine batch_size and time_steps for CNN
        x = x.view(batch_size * time_steps, channels, height, -1)
        x = self.cnn(x)

        #restore time_steps for LStm
        x = x.view(batch_size, time_steps, -1)
        x, _ = self.lstm(x)

        #get the output of the last time step for classification
        x = x[:, -1, :]
        x = self.fc(x)

        return x