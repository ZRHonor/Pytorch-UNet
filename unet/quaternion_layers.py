import torch
from torch.nn import Module, Parameter
from .quaternion_ops import quaternion_conv, quaternion_conv_rotation, \
        quaternion_transpose_conv, quaternion_tranpose_conv_rotation, \
        quaternion_linear, quaternion_linear_rotation, QuaternionLinearFunction
from .quaternion_ops import get_kernel_and_weight_shape
from numpy.random import RandomState
import numpy as np
from torch.nn.modules.utils import _pair


class UnilateralQuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    in_channels : the channel of Quaternion, which is one quarter of Real channels;
    out_channels : the channel of Quaternion, which is one quarter of Real channels;
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                dilatation=1, padding=0, groups=1, bias=True, 
                init_criterion='glorot', weight_init='unitary',
                seed=None, operation='convolution2d', rotation=False):
        super(UnilateralQuaternionConv, self).__init__()
        self.in_channels     =    in_channels
        self.out_channels    =    out_channels
        self.stride          =    stride
        self.padding         =    padding
        self.groups          =    groups
        self.dilatation      =    dilatation
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init    
        self.seed            =    seed if seed is not None else 1234
        self.rng             =    RandomState(self.seed)
        self.operation       =    operation
        self.rotation        =    rotation

        # if (self.in_channels % 4 != 0) or (self.out_channels % 4 != 0):
        #     raise Exception('Channels must be ')

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation, self.in_channels, self.out_channels, kernel_size)
        # print(self.w_shape)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels*4))
        else:
            self.register_parameter('bias', None)
        
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)
           
    def init_parameters(self):
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0, std)
                self.i_weight.data.normal_(0, std)
                self.j_weight.data.normal_(0, std)
                self.k_weight.data.normal_(0, std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.weight_init == 'quaternion':
            self.r_weight = self.modulus * torch.cos(self.phi)
            self.i_weight = self.modulus * torch.sin(self.phi) * torch.cos(self.theta)
            self.j_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.alpha)
            self.k_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.sin(self.alpha)

        if self.rotation:
            return quaternion_conv_rotation(input, self.r_weight, self.i_weight, self.j_weight, 
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight, 
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)
        
    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) \
            + ', rotation'        + str(self.rotation)    + '}'


class UnilateralQuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    in_channels : the channel of Quaternion, which is one quarter of Real channels;
    out_channels : the channel of Quaternion, which is one quarter of Real channels;
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='unitary', seed=None, operation='convolution2d', rotation=False):
        super(UnilateralQuaternionTransposeConv, self).__init__()
        self.in_channels     =    in_channels
        self.out_channels    =    out_channels
        self.stride          =    stride
        self.padding         =    padding
        self.output_padding  =    output_padding
        self.groups          =    groups
        self.dilatation      =    dilatation
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1234
        self.rng             =    RandomState(self.seed)
        self.operation       = operation
        self.rotation = rotation

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation, self.out_channels, self.in_channels, kernel_size)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels*4))
        else:
            self.register_parameter('bias', None)
        
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)

    def init_parameters(self):
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0,std)
                self.i_weight.data.normal_(0,std)
                self.j_weight.data.normal_(0,std)
                self.k_weight.data.normal_(0,std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.weight_init == 'quaternion':
            self.r_weight = self.modulus * torch.cos(self.phi)
            self.i_weight = self.modulus * torch.sin(self.phi) * torch.cos(self.theta)
            self.j_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.alpha)
            self.k_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.sin(self.alpha)
            
        if self.rotation:
            return quaternion_tranpose_conv_rotation(input, self.r_weight, self.i_weight, 
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding, 
                self.output_padding, self.groups, self.dilatation)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight, 
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding, 
                self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) \
            + ', rotation'        + str(self.rotation)    + '}'


class UnilateralQuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data. 
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='unitary',
                 seed=None, rotation=False):

        super(UnilateralQuaternionLinear, self).__init__()
        self.in_features = in_features//4
        self.out_features = out_features//4
        self.rotation = rotation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.w_shape = [self.in_features, self.out_features]

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels*4))
        else:
            self.register_parameter('bias', None)
        
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)

    def init_parameters(self):
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0,std)
                self.i_weight.data.normal_(0,std)
                self.j_weight.data.normal_(0,std)
                self.k_weight.data.normal_(0,std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.weight_init == 'quaternion':
            self.r_weight = self.modulus * torch.cos(self.phi)
            self.i_weight = self.modulus * torch.sin(self.phi) * torch.cos(self.theta)
            self.j_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.alpha)
            self.k_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.sin(self.alpha)

        if self.rotation:
            return quaternion_linear_rotation(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', rotation'        + str(self.rotation)    + '}'


class UnilateralQuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. 
    Compare to UnilateralQuaternionLinear(), this function needs less VRAM and more computing time.
    """
    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='unitary',
                 seed=None, rotation=False):

        super(UnilateralQuaternionLinearAutograd, self).__init__()
        self.in_features = in_features//4
        self.out_features = out_features//4
        self.rotation = rotation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.w_shape = [self.in_features, self.out_features]

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels*4))
        else:
            self.register_parameter('bias', None)
        
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)

    def init_parameters(self):
        # NOTE
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0,std)
                self.i_weight.data.normal_(0,std)
                self.j_weight.data.normal_(0,std)
                self.k_weight.data.normal_(0,std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.weight_init == 'quaternion':
            self.r_weight = self.modulus * torch.cos(self.phi)
            self.i_weight = self.modulus * torch.sin(self.phi) * torch.cos(self.theta)
            self.j_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.alpha)
            self.k_weight = self.modulus * torch.sin(self.phi) * torch.sin(self.theta) * torch.sin(self.alpha)
            
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) \
            + ', rotation'        + str(self.rotation)    + '}'


class BilateralQuaternionConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_criterion='glorot', weight_init='unitary', seed=None):
        super(BilateralQuaternionConv2d, self).__init__()
        self.stride = _pair(stride)
        self.kH, self.kW = _pair(kernel_size)
        self.kernel_size = kernel_size
        self.padding = _pair(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1234
        self.rng             =    RandomState(self.seed)
        self.w_shape = [self.out_channels, self.in_channels, self.kH, self.kW]
            
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)

    def init_parameters(self):
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0,std)
                self.i_weight.data.normal_(0,std)
                self.j_weight.data.normal_(0,std)
                self.k_weight.data.normal_(0,std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return self.implementation(x)

    def implementation(self, x):
        if self.weight_init == 'quaternion':
            q0 = self.modulus * torch.cos(self.phi/2)
            q1 = self.modulus * torch.sin(self.phi/2) * torch.cos(self.theta)
            q2 = self.modulus * torch.sin(self.phi/2) * torch.sin(self.theta) * torch.cos(self.alpha)
            q3 = self.modulus * torch.sin(self.phi/2) * torch.sin(self.theta) * torch.sin(self.alpha)
            s = self.modulus
        if self.weight_init == 'unitary':
            q0 = self.r_weight
            q1 = self.i_weight
            q2 = self.j_weight
            q3 = self.k_weight
            s = torch.sqrt(torch.pow(q0,2) + torch.pow(q1,2) + torch.pow(q2,2) + torch.pow(q3,2))

        modulus_2 = torch.pow(s, 2)
        r11 = modulus_2 - 2*torch.pow(q2,2) - 2*torch.pow(q3,2)
        r12 = 2*q1*q2 - 2*q0*q3
        r13 = 2*q1*q3 + 2*q0*q2
        r21 = 2*q1*q2 + 2*q0*q3
        r22 = modulus_2 - 2*torch.pow(q1,2) - 2*torch.pow(q3,2)
        r23 = 2*q2*q3 - 2*q0*q1
        r31 = 2*q1*q3 - 2*q0*q2
        r32 = 2*q2*q3 + 2*q0*q1
        r33 = modulus_2 - 2*torch.pow(q1,2) - 2*torch.pow(q2,2)
        # r** of shape (out_c, in_c, 1, kH, kW)
        r1 = torch.cat((r11*s, r12*s, r13*s), dim=1)
        r2 = torch.cat((r21*s, r22*s, r23*s), dim=1)
        r3 = torch.cat((r31*s, r32*s, r33*s), dim=1)
        R = torch.cat((r1,r2,r3), dim=0)
        #R = torch.squeeze(R,dim=2)

        output = torch.nn.functional.conv2d(x, R, stride=self.stride, padding=self.padding)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) + '}'


class BilateralQuaternionDeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_criterion='glorot', weight_init='unitary', seed=None):
        super(BilateralQuaternionDeConv2d, self).__init__()
        self.stride = _pair(stride)
        self.kH, self.kW = _pair(kernel_size)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1234
        self.rng             =    RandomState(self.seed)
        self.w_shape = [self.out_channels, self.in_channels, self.kH, self.kW]
            
        if self.weight_init == 'unitary':
            self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        elif self.weight_init == 'quaternion':
            self.modulus = Parameter(torch.Tensor(*self.w_shape))
            self.phi = Parameter(torch.Tensor(*self.w_shape))
            self.theta = Parameter(torch.Tensor(*self.w_shape))
            self.alpha = Parameter(torch.Tensor(*self.w_shape))
            self.init_parameters()
        else:
            raise Exception('Wrong weight init method. \'quaternion\' or \'unitary\'.'
                            'Input weight init method:' + self.weight_init)
    def init_parameters(self):
        # https://keras-cn.readthedocs.io/en/latest/other/initializations/
        if self.weight_init == 'unitary':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.r_weight.data.uniform_(-limit, limit)
                self.i_weight.data.uniform_(-limit, limit)
                self.j_weight.data.uniform_(-limit, limit)
                self.k_weight.data.uniform_(-limit, limit)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.r_weight.data.normal_(0,std)
                self.i_weight.data.normal_(0,std)
                self.j_weight.data.normal_(0,std)
                self.k_weight.data.normal_(0,std)
        elif self.weight_init == 'quaternion':
            if self.init_criterion == 'glorot':
                limit = np.sqrt(6 / (self.in_channels + self.out_channels))
                self.modulus.data.uniform_(-limit, limit)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)
            elif self.init_criterion == 'he':
                std = np.sqrt(2/self.in_channels)
                self.modulus.data.normal_(0,std)
                self.phi.data.uniform_(-np.pi, np.pi)
                self.theta.data.uniform_(-np.pi, np.pi)
                self.alpha.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return implementation(x)

    def implementation(self, x):
        if self.weight_init == 'quaternion':
            q0 = self.modulus * torch.cos(self.phi/2)
            q1 = self.modulus * torch.sin(self.phi/2) * torch.cos(self.theta)
            q2 = self.modulus * torch.sin(self.phi/2) * torch.sin(self.theta) * torch.cos(self.alpha)
            q3 = self.modulus * torch.sin(self.phi/2) * torch.sin(self.theta) * torch.sin(self.alpha)
            s = self.modulus
        if self.weight_init == 'unitary':
            q0 = self.r_weight
            q1 = self.i_weight
            q2 = self.j_weight
            q3 = self.k_weight
            s = torch.sqrt(torch.pow(q0,2) + torch.pow(q1,2) + torch.pow(q2,2) + torch.pow(q3,2))

        modulus_2 = torch.pow(s, 2)
        r11 = modulus_2 - 2*torch.pow(q2,2) - 2*torch.pow(q3,2)
        r12 = 2*q1*q2 - 2*q0*q3
        r13 = 2*q1*q3 + 2*q0*q2
        r21 = 2*q1*q2 + 2*q0*q3
        r22 = modulus_2 - 2*torch.pow(q1,2) - 2*torch.pow(q3,2)
        r23 = 2*q2*q3 - 2*q0*q1
        r31 = 2*q1*q3 - 2*q0*q2
        r32 = 2*q2*q3 + 2*q0*q1
        r33 = modulus_2 - 2*torch.pow(q1,2) - 2*torch.pow(q2,2)
        # r** of shape (out_c, in_c, 1, kH, kW)
        r1 = torch.cat((r11*s, r12*s, r13*s), dim=1)
        r2 = torch.cat((r21*s, r22*s, r23*s), dim=1)
        r3 = torch.cat((r31*s, r32*s, r33*s), dim=1)
        R = torch.cat((r1,r2,r3), dim=0)
        #R = torch.squeeze(R,dim=2)

        output = torch.nn.functional.conv_transpose2d(x, R, stride=self.stride, padding=self.padding, output_padding = self.output_padding)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
      
        
class QuaternionReLU(Module):
    r'''Apply Qutrnion ReLU to input.
    mode={'realReLU', 'modReLU', 'qReLU'},
    bias, float in (0,1), is needed if mode='modReLU'
    noAct: Do nothing;
    realReLU: Apply ReLU() to 4 channels of a quaternion respectively;
    modReLU: act(q) = 0,                    |q| < bias
                    = (|q|-bias)/|q| * q,   otherwise
    qReLU: act(q)   = q,    if r,i,j,k > 0
                    = 0,    otherwise
    '''
    def __init__(self, mode='realReLU', bias=None):
        super(QuaternionReLU, self).__init__()
        modes = ['realReLU', 'modReLU', 'qReLU', 'noAct']
        if not (mode in modes):
            raise Exception('Wrong ReLU mode.'
                            'Input mode: ' + mode)
        self.mode = mode
        if bias is not None:
            self.bias = bias
        self.epsilon = 1e-6
        self.act_func = {'realReLU':self.realReLU, 'modReLU':self.modReLU, 'qReLU':self.qReLU, 'noAct':self.noAct}[self.mode]


    def forward(self, x):
        return self.act_func(x)

    def noAct(self, x):
        return x

    def realReLU(self, x):
        return torch.nn.functional.relu(x)
    
    def splitQuaternionChannels(self, x):
        real_channels = x.shape[1]
        quternion_channels = real_channels // 4
        if x.device.type == 'cpu':
            r_data = x.index_select(1, torch.arange(quternion_channels))
            i_data = x.index_select(1, torch.arange(quternion_channels, 2*quternion_channels))
            j_data = x.index_select(1, torch.arange(2*quternion_channels, 3*quternion_channels))
            k_data = x.index_select(1, torch.arange(3*quternion_channels, 4*quternion_channels))
        else:
            r_data = x.index_select(1, torch.arange(quternion_channels).cuda())
            i_data = x.index_select(1, torch.arange(quternion_channels, 2*quternion_channels).cuda())
            j_data = x.index_select(1, torch.arange(2*quternion_channels, 3*quternion_channels).cuda())
            k_data = x.index_select(1, torch.arange(3*quternion_channels, 4*quternion_channels).cuda())
        return r_data, i_data, j_data, k_data

    def modReLU(self, x):
        (r_data, i_data, j_data, k_data) = self.splitQuaternionChannels(x)
        mod = torch.sqrt(torch.pow(r_data, 2) + torch.pow(i_data, 2) + torch.pow(j_data, 2) + torch.pow(k_data, 2))
        act = mod - self.bias
        mask = act.ge(0)
        mask = mask.float()
        act = torch.mul(mask, act)
        act_mod = torch.div(act+self.epsilon, mod+self.epsilon)
        return torch.cat((torch.mul(r_data, act_mod), torch.mul(i_data, act_mod), torch.mul(j_data, act_mod), torch.mul(k_data, act_mod)), dim=1)

    def qReLU(self, x):
        (r_data, i_data, j_data, k_data) = self.splitQuaternionChannels(x)
        mask = r_data.ge(0) * i_data.ge(0) * j_data.ge(0) * k_data.ge(0)
        mask = mask.float()
        # mask = torch.FloatTensor(mask)
        # temp = r_data*mask
        # print("temp"+str(temp.type()))
        # temp1 = torch.mul(r_data,mask)
        # print("temp1"+str(temp1.type()))
        # temp2 = torch.cat((torch.mul(r_data, mask), torch.mul(i_data, mask), torch.mul(j_data, mask), torch.mul(k_data, mask)), dim=1)
        # print("temp2"+str(temp2.type()))
        return torch.cat((torch.mul(r_data, mask), torch.mul(i_data, mask), torch.mul(j_data, mask), torch.mul(k_data, mask)), dim=1)

    def __repr__(self):
        if self.mode == 'modReLU':
            return self.__class__.__name__ + '{' \
                + 'mode='   +str(self.mode) \
                +', bias='  +str(self.bias) +'}'
        else:
            return self.__class__.__name__ + '{' \
                + 'mode='   +str(self.mode) +'}'


class QuaternionBN(Module):
    # TODO QuaternionBN
    def __init__(self, mode='rBN'):
        super(QuaternionBN, self).__init__()
        modes = ['rBN', 'qBN']
        if mode in modes:
            self.mode = modes
        else:
             raise Exception('Wrong Batch Normalization mode.'
                            'Input mode: ' + mode)
        self.BN_func = {'rBN':self.rBN, 'qBN':self.qBN}[self.mode]
    
    def rBN(self, x):
        return torch.nn.functional.batch_norm(x)

    def splitQuaternionChannels(self, x):
        real_channels = x.shape[1]
        quternion_channels = real_channels // 4
        if x.device.type == 'cpu':
            r_data = x.index_select(1, torch.arange(quternion_channels))
            i_data = x.index_select(1, torch.arange(quternion_channels, 2*quternion_channels))
            j_data = x.index_select(1, torch.arange(2*quternion_channels, 3*quternion_channels))
            k_data = x.index_select(1, torch.arange(3*quternion_channels, 4*quternion_channels))
        else:
            r_data = x.index_select(1, torch.arange(quternion_channels).cuda())
            i_data = x.index_select(1, torch.arange(quternion_channels, 2*quternion_channels).cuda())
            j_data = x.index_select(1, torch.arange(2*quternion_channels, 3*quternion_channels).cuda())
            k_data = x.index_select(1, torch.arange(3*quternion_channels, 4*quternion_channels).cuda())
        return r_data, i_data, j_data, k_data


    def qBN(self, x):
        r'''https://arxiv.org/pdf/1705.09792.pdf
        '''
        r_data, i_data, j_data, k_data = self.splitQuaternionChannels(x)
        channels = r_data.shape[1]
        covs = torch.nn.Parameter(torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
        # NOTE covariance matrix =
        # [covs[0],   covs[4],    covs[5],    covs[6];
        #  covs[4],   covs[1],    covs[7],    covs[8];
        #  covs[5],   covs[7],    covs[2],    covs[9];
        #  covs[6],   covs[8],    covs[9],    covs[3]]
        # only 10 parameters
        r_BN = covs[0]*r_data + covs[4]*i_data + covs[5]*j_data + covs[6]*k_data
        i_BN = covs[4]*r_data + covs[1]*i_data + covs[7]*j_data + covs[8]*k_data
        j_BN = covs[5]*r_data + covs[7]*i_data + covs[2]*j_data + covs[9]*k_data
        k_BN = covs[6]*r_data + covs[8]*i_data + covs[9]*j_data + covs[3]*k_data
        return torch.cat((r_BN, i_BN, j_BN, k_BN), dim=1)

    def forward(self, x):
        return self.BN_func(x)

    def __repr__(self):
        return self.__class__.__name__ + '{' \
            + 'mode='   +str(self.mode) +'}'