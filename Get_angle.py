import initialise
import functions as funcs
import wave
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time
import math
def run(filename,tone_freq):
    az_final_ang = 0
    el_final_ang = 0
    fs = 44100


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

    """micR1_nums = [8,7,10,9]
    micR2_nums = [6,5,12,11]
    micR3_nums = [4,3,14,13]
    micR4_nums = [2,1,16,15]"""
    micR1_nums = [9,10,7,8]
    micR2_nums = [11,12,5,6]
    micR3_nums = [13,14,3,4]
    micR4_nums = [15,16,1,2]

    micC1_nums = [8,6,4,2]
    micC2_nums = [7,5,3,1]
    micC3_nums = [10,12,14,16]
    micC4_nums = [9,11,13,15]

    micsC1 = [mic8,mic6,mic4,mic2]
    micsC2 = [mic7,mic5,mic3,mic1]
    micsC3 = [mic10,mic12,mic14,mic16]
    micsC4 = [mic9,mic11,mic13,mic15]

    inChannels = funcs.get_channel_data(filename)

    ambiguities = funcs.determine_baseline(tone_freq)[0]
    if ambiguities == True:
        az_angles = np.zeros((4,1),dtype=np.float32)

        az_angles[0], _ = funcs.rattle_off_phase_diffs(inChannels,micR1_nums,tone_freq,micsR1,fs)
        az_angles[1], _ = funcs.rattle_off_phase_diffs(inChannels, micR2_nums, tone_freq, micsR2, fs)
        az_angles[2], _ = funcs.rattle_off_phase_diffs(inChannels,micR3_nums,tone_freq,micsR3,fs)
        az_angles[3], _ = funcs.rattle_off_phase_diffs(inChannels,micR4_nums,tone_freq,micsR4,fs)
        az_angles = funcs.remove_outliers(az_angles)
        az_final_ang = sum(az_angles)/len(az_angles)
    else:
        az_angles = []

        az_angle_array, _ = funcs.rattle_off_phase_diffs(inChannels, micR1_nums, tone_freq, micsR1, fs)
        az_angles.append(az_angle_array)
        az_angle_array, _ = funcs.rattle_off_phase_diffs(inChannels, micR2_nums, tone_freq, micsR2, fs)
        az_angles.append(az_angle_array)
        az_angle_array, _ = funcs.rattle_off_phase_diffs(inChannels, micR3_nums, tone_freq, micsR3, fs)
        az_angles.append(az_angle_array)
        az_angle_array, _ = funcs.rattle_off_phase_diffs(inChannels, micR4_nums, tone_freq, micsR4, fs)
        az_angles.append(az_angle_array)

        likely_az_angles = funcs.make_array_of_likely_angles(az_angles)
        snrs = funcs.find_snrs(inChannels, likely_az_angles, tone_freq)
        az_final_ang = [funcs.get_angle_from_snrs(snrs, likely_az_angles)]

    if ambiguities == True:
        el_angles = np.zeros((4, 1), dtype=np.float32)

        el_angles[0], _ = funcs.rattle_off_phase_diffs(inChannels, micC1_nums, tone_freq, micsC1, fs,axis="Column")
        el_angles[1], _ = funcs.rattle_off_phase_diffs(inChannels, micC2_nums, tone_freq, micsC2, fs,axis="Column")
        el_angles[2], _ = funcs.rattle_off_phase_diffs(inChannels, micC3_nums, tone_freq, micsC3, fs, axis="Column")
        el_angles[3], _ = funcs.rattle_off_phase_diffs(inChannels, micC4_nums, tone_freq, micsC4, fs,axis="Column")
        el_angles = funcs.remove_outliers(el_angles)
        el_final_ang = sum(el_angles) / len(el_angles)

        """print("az = " + str(az_final_ang))
        print("el = " + str(el_final_ang))"""

    else:
        el_angles = []
        el_angle_array, ambiguities = funcs.rattle_off_phase_diffs(inChannels, micC1_nums, tone_freq, micsC1, fs,axis="Column")
        el_angles.append(el_angle_array)
        el_angle_array, ambiguities = funcs.rattle_off_phase_diffs(inChannels, micC2_nums, tone_freq, micsC2, fs,axis="Column")
        el_angles.append(el_angle_array)
        el_angle_array, ambiguities = funcs.rattle_off_phase_diffs(inChannels, micC3_nums, tone_freq, micsC3, fs, axis="Column")
        el_angles.append(el_angle_array)
        el_angle_array, ambiguities = funcs.rattle_off_phase_diffs(inChannels, micC4_nums, tone_freq, micsC4, fs,axis="Column")
        el_angles.append(el_angle_array)
        likely_el_angles = funcs.make_array_of_likely_angles(el_angles)
        temp_hard = likely_el_angles[0]
        for angle in likely_el_angles:
            if np.abs(angle-90) < np.abs(temp_hard-90):
                temp_hard = angle

        snrs = funcs.find_snrs(inChannels, likely_el_angles, tone_freq)
        #el_final_ang = [funcs.get_angle_from_snrs(snrs, likely_el_angles)]
        el_final_ang = [temp_hard]

        """print("azimuth angle = " + str(az_angle))
        print("elevation angle = " + str(el_final_angle))"""

    return [str(az_final_ang[0]),str(el_final_ang[0])]