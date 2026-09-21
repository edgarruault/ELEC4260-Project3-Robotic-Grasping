import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspCNN(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None):
        """
        A convolutional neural network for grasp prediction.
        This network predicts grasp positions, orientations (using cosine and sine components),
        and widths from input images. It uses a series of convolutional layers with downsampling
        and upsampling to process the input image.
        """
        super().__init__()

        # TODO: Initialize layers and components of GraspCNN (Task3)
        if filter_sizes is None:
            filter_sizes = [32, 64, 128, 64, 32]
            
        f1, f2, f3 , f4, f5 = filter_sizes

        #downsampling
        self.conv1 = nn.Conv2d(input_channels, f1, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=3, stride=2, padding=1)

        #bottleneck
        self.conv4 = nn.Conv2d(f3, f3, kernel_size=3, stride=1, padding=1)

        #upsampling
        self.deconv1 = nn.ConvTranspose2d(f3, f4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(f4, f5, kernel_size=4, stride=2, padding=1)

        #output
        self.pos_output = nn.Conv2d(f5, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(f5, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(f5, 1, kernel_size=1)
        self.width_output = nn.Conv2d(f5, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            tuple: A tuple containing four tensors:
                - pos_output: Grasp position predictions
                - cos_output: Cosine of grasp angle predictions
                - sin_output: Sine of grasp angle predictions
                - width_output: Grasp width predictions
        """
        # TODO: Implement GraspCNN's forward pass for inference (Task3)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        """
        Compute the loss for the network predictions.
        
        Args:
            xc (torch.Tensor): Input tensor
            yc (tuple): Ground truth tuple containing:
                - y_pos: Ground truth grasp positions
                - y_cos: Ground truth cosine components
                - y_sin: Ground truth sine components
                - y_width: Ground truth grasp widths
                
        Returns:
            dict: A dictionary containing:
                - 'loss': Total combined loss
                - 'losses': Dictionary of individual loss components
                - 'pred': Dictionary of model predictions
        """
        # TODO: Implement GraspCNN's loss function (Task3)
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)


        pos_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        total_loss = pos_loss + cos_loss + sin_loss + width_loss

        return {
            'loss':total_loss,
            'losses': {
                'pos_loss':pos_loss,
                'cos_loss':cos_loss,
                'sin_loss':sin_loss,
                'width_loss':width_loss
            },
            'pred': {
                'pos':pos_pred,
                'cos':cos_pred,
                'sin':sin_pred,
                'width':width_pred
            }
        }
