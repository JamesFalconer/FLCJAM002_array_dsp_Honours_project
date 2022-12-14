import numpy as np
import sympy as sym
import math
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import functions as funcs
import time


def run(filename, null_ang,tone_freq):
    null_ang = null_ang
    fs = 44100

    ang = null_ang * np.pi / 180


    mic1 = [0.021,-0.063,0]
    mic2 = [0.063,-0.063,0]
    mic3 = [0.021,-0.021,0]
    mic4 = [0.063,-0.021,0]
    mic5 = [0.021,0.021,0]
    mic6 = [0.063,0.021,0]
    mic7 = [0.021,0.063,0]
    mic8 = [0.063,0.063,0]
    mic9 = [-0.063,0.063,0]
    mic10 = [-0.021,0.063,0]
    mic11 = [-0.063,0.021,0]
    mic12 = [-0.021,0.021,0]
    mic13 = [-0.063,-0.021,0]
    mic14 = [-0.021,-0.021,0]
    mic15 = [-0.063,-0.063,0]
    mic16 = [-0.021,-0.063,0]

    micsR1 = [mic8,mic7,mic10,mic9]
    micsR2 = [mic6,mic5,mic12,mic11]
    micsR3 = [mic4,mic3,mic14,mic13]
    micsR4 = [mic2,mic1,mic16,mic15]
    micCols = [micsR1,micsR2,micsR3,micsR4]

    """ micsR1 = [mic9, mic10, mic7, mic8]
    micsR2 = [mic11, mic12, mic5, mic6]
    micsR3 = [mic13, mic14, mic3, mic4]
    micsR4 = [mic15, mic16, mic1, mic2]
    micCols = [micsR1, micsR2, micsR3, micsR4]"""
    micR1_nums = [8,7,10,9]
    micR2_nums = [6,5,12,11]
    micR3_nums = [4,3,14,13]
    micR4_nums = [2,1,16,15]
    """micR1_nums = [9, 10, 7, 8]
    micR2_nums = [11, 12, 5, 6]
    micR3_nums = [13, 14, 3, 4]
    micR4_nums = [15, 16, 1, 2]"""
    micCols_num = [micR1_nums, micR2_nums, micR3_nums, micR4_nums]

    micC1_nums = [8,6,4,2]
    micC2_nums = [7,5,3,1]
    micC3_nums = [10,12,14,16]
    micC4_nums = [9,11,13,15]

    micRows_num = [micC1_nums, micC2_nums, micC3_nums, micC4_nums]

    inChannels = funcs.get_channel_data(filename)

    st = time.time()
    xf = fftfreq(len(inChannels[1]), 1 / fs)
    weights = funcs.get_null_weights_y(len(xf), xf, ang, 1)

    weights = np.round(weights,9)
    temp = np.zeros((16, len(inChannels[1])),dtype=complex)
    for i in range(4):
        for p in range(4):
            temp[4*i + p,:] = inChannels[micRows_num[i][p]-1,:]

    shifted = np.zeros((16, len(inChannels[1])))
    temp1 = np.zeros((16,len(inChannels[0])),dtype=complex)

    for i in range(4):
        for p in range(4):
            X = fft(temp[4*i + p,:])
            xf = fftfreq(len(temp[i]), 1/fs)
            weighted = np.zeros(len(temp[i]), dtype=complex)

            weighted = funcs.apply_weighting(weights[p], X)

            shift = ifft(weighted)

            temp1[4*i + p] = shift

    final = funcs.add_sixteen_channels(temp1)

    s0 = funcs.snr(inChannels[0],tone_freq)
    s1 = funcs.snr(funcs.add_sixteen_channels(inChannels),tone_freq)
    s = funcs.snr(final,tone_freq)
    print("For a null steer angle of " + str(null_ang) + " elevation")
    print("SNR for a single Channel = "+str(s0))
    print("SNR for combined Channels = "+str(s1))
    print("SNR for beamformed Channels = "+str(s))


