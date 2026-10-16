# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Copyright (c) 2021, NVIDIA Corporation & affiliates. All rights reserved.

NVIDIA Source Code License for StyleGAN3

=======================================================================

1. Definitions

"Licensor" means any person or entity that distributes its Work.

"Software" means the original work of authorship made available under
this License.

"Work" means the Software and any additions to or derivative works of
the Software that are made available under this License.

The terms "reproduce," "reproduction," "derivative works," and
"distribution" have the meaning as provided under U.S. copyright law;
provided, however, that for the purposes of this License, derivative
works shall not include works that remain separable from, or merely
link (or bind by name) to the interfaces of, the Work.

Works, including the Software, are "made available" under this License
by including in or with the Work either (a) a copyright notice
referencing the applicability of this License to the Work, or (b) a
copy of this License.

2. License Grants

    2.1 Copyright Grant. Subject to the terms and conditions of this
    License, each Licensor grants to you a perpetual, worldwide,
    non-exclusive, royalty-free, copyright license to reproduce,
    prepare derivative works of, publicly display, publicly perform,
    sublicense and distribute its Work and any resulting derivative
    works in any form.

3. Limitations

    3.1 Redistribution. You may reproduce or distribute the Work only
    if (a) you do so under this License, (b) you include a complete
    copy of this License with your distribution, and (c) you retain
    without modification any copyright, patent, trademark, or
    attribution notices that are present in the Work.

    3.2 Derivative Works. You may specify that additional or different
    terms apply to the use, reproduction, and distribution of your
    derivative works of the Work ("Your Terms") only if (a) Your Terms
    provide that the use limitation in Section 3.3 applies to your
    derivative works, and (b) you identify the specific derivative
    works that are subject to Your Terms. Notwithstanding Your Terms,
    this License (including the redistribution requirements in Section
    3.1) will continue to apply to the Work itself.

    3.3 Use Limitation. The Work and any derivative works thereof only
    may be used or intended for use non-commercially. Notwithstanding
    the foregoing, NVIDIA and its affiliates may use the Work and any
    derivative works commercially. As used herein, "non-commercially"
    means for research or evaluation purposes only.

    3.4 Patent Claims. If you bring or threaten to bring a patent claim
    against any Licensor (including any claim, cross-claim or
    counterclaim in a lawsuit) to enforce any patents that you allege
    are infringed by any Work, then your rights under this License from
    such Licensor (including the grant in Section 2.1) will terminate
    immediately.

    3.5 Trademarks. This License does not grant any rights to use any
    Licensor’s or its affiliates’ names, logos, or trademarks, except
    as necessary to reproduce the notices described in this License.

    3.6 Termination. If you violate any term of this License, then your
    rights under this License (including the grant in Section 2.1) will
    terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
THIS LICENSE.

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
(INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES.

Generator architecture from the paper
"Alias-Free Generative Adversarial Networks".

=======================================================================
"""
# The code is borrowed for Research purposes only


import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch_utils import persistence

import torch.nn.utils.parametrize as parametrize
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn


class LReLu_regular(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,  # Input spatial size: int or [width, height].
                 out_size,  # Output spatial size: int or [width, height].
                 in_sampling_rate,  # Input sampling rate (s).
                 out_sampling_rate,  # Output sampling rate (s).
                 ):
        super().__init__()

        self.activation = nn.LeakyReLU()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate


    def forward(self, x):

        if self.in_sampling_rate == 2 * self.out_sampling_rate:
            return nn.AvgPool2d(2, stride=2, padding=0)(self.activation(x))
        elif self.in_sampling_rate == 4 * self.out_sampling_rate:
            return nn.AvgPool2d(4, stride=4, padding=1)(self.activation(x))
        else:
            return nn.functional.interpolate(self.activation(x), size=self.out_size)


class LReLu_torch(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,  # Input spatial size: int or [width, height].
                 out_size,  # Output spatial size: int or [width, height].
                 in_sampling_rate,  # Input sampling rate (s).
                 out_sampling_rate,  # Output sampling rate (s).
                 ):
        super().__init__()

        self.activation = nn.LeakyReLU()

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert in_channels == out_channels

        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.mode = 'bilinear'

    def forward(self, x):
        x = nn.functional.interpolate(x, size=2 * self.in_size, mode=self.mode, antialias=True)
        x = self.activation(x)
        x = nn.functional.interpolate(x, size=self.in_size, mode=self.mode, antialias=True)
        if self.in_size != self.out_size:
            x = nn.functional.interpolate(x, size=self.out_size, mode=self.mode, antialias=True)

        x = x.permute(0, 2, 3, 1)
        x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
        x = x.permute(0, 3, 1, 2)
        return x
