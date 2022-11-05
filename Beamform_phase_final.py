import functions as funcs
import wave
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

def run(filename,az_angle,el_angle,tone_freq):

    c = 343  # speed of sound
    fs = 44100  # sampling frequency

    mic1 = [0.021,-0.063]
    mic2 = [0.063,-0.063]
    mic3 = [0.021,-0.021]
    mic4 = [0.063,-0.021]
    mic5 = [0.021,0.021]
    mic6 = [0.063,0.021]
    mic7 = [0.021,0.063]
    mic8 = [0.063,0.063]
    mic9 = [-0.063,0.063]
    mic10 = [-0.021,0.063]
    mic11 = [-0.063,0.021]
    mic12 = [-0.021,0.021]
    mic13 = [-0.063,-0.021]
    mic14 = [-0.021,-0.021]
    mic15 = [-0.063,-0.063]
    mic16 = [-0.021,-0.063]


    """micsR1 = [mic8,mic7,mic10,mic9]
    micsR2 = [mic6,mic5,mic12,mic11]
    micsR3 = [mic4,mic3,mic14,mic13]
    micsR4 = [mic2,mic1,mic16,mic15]
    micCols = [micsR1,micsR2,micsR3,micsR4]
    """
    micsR1 = [mic9, mic10, mic7, mic8]
    micsR2 = [mic11, mic12, mic5, mic6]
    micsR3 = [mic13, mic14, mic3, mic4]
    micsR4 = [mic15, mic16, mic1, mic2]
    micCols = [micsR1, micsR2, micsR3, micsR4]

    """micR1_nums = [8,7,10,9]
    micR2_nums = [6,5,12,11]
    micR3_nums = [4,3,14,13]
    micR4_nums = [2,1,16,15]"""
    micR1_nums = [9,10,7,8]
    micR2_nums = [11,12,5,6]
    micR3_nums = [13,14,3,4]
    micR4_nums = [15,16,1,2]
    micCols_num = [micR1_nums, micR2_nums, micR3_nums, micR4_nums]


    inChannels = funcs.get_channel_data(filename)

    times = []
    for i in range(4):
        temp = funcs.delay_time(micCols[i],az_angle,el_angle,c)
        for l in range(4):
            times.append(temp[l])


    shifted = []

    st = time.time()
    for i in range(4):
        for p in range(4):
            shifted.append(funcs.delay_by_rotation(inChannels[micCols_num[i][p]-1], fs, times[4*i + p]))
    total = funcs.add_sixteen_channels(shifted)

    s0 = funcs.snr(inChannels[0], tone_freq)
    s1 = funcs.snr(funcs.add_sixteen_channels(inChannels), tone_freq)
    s = funcs.snr(total, tone_freq)
    print("For steer angle of " + str(az_angle) + " azimuth and " + str(el_angle) + " elevation:")
    print("SNR for a single Channel = " + str(s0))
    print("SNR for combined Channels = " + str(s1))
    print("SNR for beamformed Channels = " + str(s))
    print()

