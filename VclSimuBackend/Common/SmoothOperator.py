'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''

"""
Do smooth operation by scipy
support: gaussian filter, and butter-worth filter
"""
import enum
import numpy as np
from scipy.ndimage import filters
from scipy import signal
from typing import Any, Dict, Optional, Union


# for enum.IntEnum, InvDynSmoothMode.GAUSSIAN == 1 is True
class SmoothMode(enum.IntEnum):
    NO = 0  # not using smooth
    GAUSSIAN = 1  # use gaussian smooth
    BUTTER_WORTH = 2  # use butter worth smooth


class GaussianBase:
    __slots__ = ("width",)

    def __init__(self, width: Optional[int]):
        self.width: Optional[int] = width


class FilterInfoBase:

    __slots__ = ("order", "wn")

    def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
        self.order = order
        self.wn = self.calc_freq(cut_off_freq, sample_freq)

    @classmethod
    def build_from_dict(cls, info: Optional[Dict[str, Any]], sample_freq: int):
        return cls(info["order"], info["cut_off_freq"], sample_freq) if info is not None else None

    @staticmethod
    def calc_freq(cut_off_freq: float, sample_freq: float) -> float:
        return cut_off_freq / (sample_freq / 2)


class ButterWorthBase(FilterInfoBase):
    __slots__ = ("order", "wn")

    def __init__(self, order: int, cut_off_freq: float, sample_freq: int):
        super(ButterWorthBase, self).__init__(order, cut_off_freq, sample_freq)


def smooth_operator(x: np.ndarray, smooth_type: Union[GaussianBase, ButterWorthBase, None]) -> np.ndarray:
    """
    The first dimension of x is time
    """
    # print(f"call smoother operator, smooth_type == {type(smooth_type)}")
    if smooth_type is None:
        result = x
    elif isinstance(smooth_type, GaussianBase):
        if smooth_type.width is not None:
            result = filters.gaussian_filter1d(x, smooth_type.width, axis=0, mode='nearest')
        else:
            result = x
    elif isinstance(smooth_type, ButterWorthBase):
        b, a = signal.butter(smooth_type.order, smooth_type.wn)
        result = signal.filtfilt(b, a, x, axis=0)
    else:
        raise NotImplementedError("Only support GaussianBase and ButterWorthBase.")

    return result


def test_butterworth_1d():
    import matplotlib.pyplot as plt
    # https://blog.csdn.net/weixin_41521681/article/details/108262389
    fs = 1000  # Sampling frequency
    t = np.arange(1000) / fs
    signala = np.sin(2 * np.pi * 100 * t)  # with frequency of 100
    signalb = np.sin(2 * np.pi * 20 * t)  # frequency 20

    signalc = signala + signalb
    plt.plot(t, signalc, label='c')

    fc = 30  # Cut-off frequency of the filter
    smooth_type = ButterWorthBase(5, fc / (fs / 2))  # # Normalize the frequency
    output = smooth_operator(signalc, smooth_type)
    plt.plot(t, output, label='filtered')
    plt.legend()
    plt.show()


def test_butterworth_2d():
    import matplotlib.pyplot as plt
    t = np.arange(1000) / 1000
    xa = np.cos(2 * np.pi * 20 * t)
    xnoise = 0.1 * np.random.random(1000) + np.sin(2 * np.pi * 100 * t)
    x = xa + xnoise

    ya = np.sin(2 * np.pi * 30 * t)
    ynoise = 0.2 * np.random.random(1000) + np.sin(2 * np.pi * 130 * t)
    y = ya + ynoise

    data = np.concatenate([x[..., None], y[..., None]], axis=1)
    smooth_type = ButterWorthBase(5, 30 / (1000 / 2))
    output = smooth_operator(data, smooth_type)
    print(output.shape)

    plt.subplot(211)
    plt.plot(t, x, color="g")
    plt.plot(t, output[:, 0], color="r")

    plt.subplot(212)
    plt.plot(t, y, color="g")
    plt.plot(t, output[:, 1], color="r")

    plt.show()


if __name__ == "__main__":
    test_butterworth_2d()
