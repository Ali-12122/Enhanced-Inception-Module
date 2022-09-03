import torch
import torch.nn as nn


# this function is NOT to be the direct implementation of the inception module,
# but only as a basic structure to modify and add on, such that we construct an
# efficient module that is to be used as a normal layer/module would.

def enhanced_inception(input_data, max_kernel_size, dimensions_of_convolution):
    # Defining variables and operations,
    #
    # convolution: Convolution operation as appropriate to the dimensions of the data.
    # max_pool: Max pooling operation as appropriate to the dimensions of the data.
    # convolution_1x1: 1x1 Convolution operation as appropriate to the dimensions of the data.
    #
    # The data is assumed to be two dimensional unless otherwise is specified.

    convolution = nn.Conv2d(input_data.size(dim=2), out_channels=64, kernel_size=(2,))
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    convolution_1x1 = nn.Conv2d(input_data.size(dim=2), out_channels=64, kernel_size=(1,))

    # A sequence of if conditions to adjust the operations of the module to the dimensions of the data.
    # The operations modified are convolution, max_pool and convolution 1x1.

    if dimensions_of_convolution == 1:
        convolution = nn.Conv1d(input_data.size(dim=2), out_channels=64, kernel_size=(2,))
        max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        convolution_1x1 = nn.Conv1d(input_data.size(dim=2), out_channels=64, kernel_size=(1,))

    elif dimensions_of_convolution == 3:
        convolution = nn.Conv3d(input_data.size(dim=2), out_channels=64, kernel_size=(2,))
        max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        convolution_1x1 = nn.Conv3d(input_data.size(dim=2), out_channels=64, kernel_size=(1,))

    # an else statement to handle invalid dimensions, if the variable dimensions_of_convolution
    # is not equal to 1 or 2 or 3.

    else:
        convolution_dimensional_error = "Invalid convolution dimensions."
        return convolution_dimensional_error

    # The Max pool branch from the inception module (future reminder: put a link to the enhanced inception here)

    convolution_input = convolution_1x1(input_data)
    collective_data = convolution_1x1(input_data)

    # Preparing the data for the iteration of convolution operation.

    max_pool_output = max_pool(input_data)
    convolution_1x1_output = convolution_1x1(max_pool_output)
    torch.cat((collective_data, convolution_1x1_output), 2)

    # Iterating the convolution operation over the data (max_kernel_size - 1) times,
    # Explanation:
    #
    # Tensor height/width after convolution = (height/width before - kernel height/width)/ stride + 1
    # more concisely H/W_out = (H/W_in _ H/W_kernel)/s +1
    # at 


    for i in range(max_kernel_size - 1):
        convolution_output = convolution(convolution_input)
        torch.cat((collective_data, convolution_output), 2)
        convolution_input = convolution_output

    return
