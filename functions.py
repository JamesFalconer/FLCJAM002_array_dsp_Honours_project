import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch
import time
from scipy import signal
from statistics import mode
import pyaudio
import wave


def delay_time_x(mics_pos, angle,speed_of_sound):
    timeToDelay = []
    angle = angle * math.pi / 180
    for i in mics_pos:
        t = i[0]*math.cos(angle)/speed_of_sound
        timeToDelay.append(t)
    return timeToDelay


def delay_time_y(mics_pos, angle,speed_of_sound):
    timeToDelay = []
    angle = angle * math.pi / 180
    for i in mics_pos:
        t = i[1]*math.sin(angle)/speed_of_sound
        timeToDelay.append(t)
    return timeToDelay


def delay_filter(delay, length):
    filterLength = length
    centreTap = filterLength / 2

    sinc1 = []
    window1 = []
    tapweight1 = []

    for i in range(filterLength):
        x = i - delay

        sinc = math.sin(math.pi * (x - centreTap)) / (math.pi * (x - centreTap))

        window = 0.54 - 0.46 * math.cos(2.0 * math.pi * (x + 0.5) / filterLength)

        tapWeight = window * sinc


        sinc1.append(sinc)
        window1.append(window)
        tapweight1.append(tapWeight)

    return tapweight1


def samples_to_delay(delayTime,samplingFrequency,select = 'both'):
    delayTime = float(delayTime)
    samplingFrequency = float(samplingFrequency)
    sampleLength = 1/samplingFrequency
    samplesToDelay = int(delayTime/sampleLength)
    remainingTime = delayTime/sampleLength - samplesToDelay

    if select == 'both':
        return samplesToDelay,remainingTime
    else:
        return samplesToDelay


def delay_signal(signal,samples_to_delay,min,max):
    m1 = int(samples_to_delay)
    m2 = int(samples_to_delay)
    temp_signal = list(signal)
    while m1<max:
        temp_signal.append(0)
        m1+=1

    while m2 > min:
        temp_signal.insert(0,0)
        m2-=1

    return temp_signal


def add_four_channels(channels):
    channelAve = []
    incr = 0
    smallest = min([len(channels[0]),len(channels[1]),len(channels[2]),len(channels[3])])
    while incr != smallest:
        ave = channels[0][incr] + channels[1][incr] + channels[2][incr] + channels[3][incr]
        ave = ave / 4
        channelAve.append(complex(ave))
        incr += 1

    channelAve = np.array(channelAve)
    return channelAve


def add_sixteen_channels(channels):
    channels = np.array(channels)
    chan_len = len(channels[0])
    channelAve = np.zeros(chan_len, dtype=complex)
    channel_lengths = []
    for i in range(chan_len):
        channelAve[i] = complex(sum(channels[:, i])/16)
    return channelAve


def wave_unit_vector(az_angle, el_angle):
    az_angle = az_angle * np.pi / 180
    el_angle = el_angle * np.pi / 180
    wx = math.cos(el_angle)*math.cos(az_angle)
    wy = math.sin(el_angle)
    wz = math.cos(el_angle)*math.sin(az_angle)

    return np.array([wx, wy, wz])


def delay_time(mics_pos, az, el, c):

    times = []
    wav = wave_unit_vector(az, el)
    for i in mics_pos:
        dist = i[0]*wav[0]+i[1]*wav[1]

        t = dist/c
        times.append(t)
    return times


def delay_by_rotation(signal, fs, t0):
    N = len(signal)
    X = fft(signal)
    freq = fftfreq(N, 1 / fs)
    a = np.zeros((len(X)), dtype=complex)
    for i in range(N):
        X[i] = X[i] * np.exp(-1j * 2 * math.pi * freq[i] * t0)

    xf = ifft(X)
    xf = np.real(xf)
    xf = xf.astype(np.int16)
    return xf


def peak_value_at_frequency(fftSignal, frequencyAxis, frequency, calibration):
    x = np.where(frequencyAxis == int(frequency))
    temp_list = fftSignal[frequency-calibration:frequency+calibration+1]
    peakValue = max(temp_list)

    return peakValue


def find_index(signal, f, frequency, calibration, show=True):
    """ finds the highest value in an array for a specific frequency by searching for the max value
     within the calibration value of samples in the given array."""
    scale = (len(f)//2)/max(f)
    signal = np.abs(signal)
    temp_frequency_index = int(scale * frequency)
    frequency_index = np.int(temp_frequency_index)
    for i in range(calibration):
        if signal[temp_frequency_index - calibration//2 + i] > signal[frequency_index]:
            frequency_index = temp_frequency_index - calibration//2 + i
    return frequency_index


def null_signal(signal, index_to_null):
    nulled_signal = np.array(signal)

    nulled_signal[index_to_null] = 0
    return nulled_signal


def delay_sum_integrate(signal, f):

    tot = sum(signal)
    tot = tot/len(f)
    print("noise power is :" + str(tot))

    return tot


def delay_sum_sig_power(signal, f):
    tot = signal
    print("signal power is :" + str(tot))
    return tot


def delay_sum_snr(signal, signal_frequency, fs=44100, freq_bin_size=100):
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal) // 2, return_onesided=False)
    plt.figure()
    plt.plot(f,Pxx)
    signal_index = find_index(Pxx, f, signal_frequency, freq_bin_size)
    Pxx_less_signal = np.array(null_signal(Pxx,signal_index))
    noise_power = np.float32(delay_sum_integrate(Pxx_less_signal[0:len(f)//2], f[0:len(f)//2]))
    signal_power = np.float32(delay_sum_sig_power(Pxx[signal_index],f))
    SNR = signal_power/noise_power
    SNR = 10 * np.log10(SNR)
    return SNR

def integrate(signal, f):

    tot = 0
    for i in signal:
        tot += np.abs(i)

    tot = tot / len(f)
    return tot


def sig_power(signal, f):
    df = len(f) / 2 / max(f)
    tot = signal
    return tot


def snr(signal, signal_frequency, fs = 44100, freq_bin_size = 30):
    f, Pxx = welch(signal, fs=fs, nperseg=len(signal) // 2, return_onesided = False)
    """    plt.figure()
    plt.plot(f,Pxx)
    plt.show()"""
    signal_index = find_index(Pxx, f, signal_frequency, freq_bin_size)
    Pxx_less_signal = null_signal(Pxx,signal_index)

    noise_power = integrate(Pxx_less_signal[0:len(f)//2], f[0:len(f)//2])
    signal_power = sig_power(Pxx[signal_index],f)
    SNR = signal_power/noise_power
    SNR = 10 * np.log10(SNR)
    return SNR


def steering_vector(mics_positions, az_angle, el_angle, num_mics, freq_len=44100, c=343):
    f = 3000
    lamda = c/f

    wave_vector = np.array(wave_unit_vector(az_angle, el_angle))
    steeringVector = np.zeros((num_mics,1),dtype = np.complex)
    for i in range(num_mics):
        r = np.array(mics_positions[i])
        steeringVector[i] = (np.exp(-1j*2*np.pi/lamda * wave_vector.dot(r)))
    steeringVectorNormalised = normalise_old_dont_use(steeringVector,freq_len)
    return steeringVectorNormalised


def normalise(steeringVector,freq_len):
    temp_weight = np.matmul(np.conjugate(steeringVector[:,0:10]).T,steeringVector[:,0:10])
    weight = max(temp_weight[0])

    steeringVector = steeringVector / weight

    return steeringVector


def normalise_old_dont_use(steeringVector,freq_len):

    weight = np.matmul(np.conjugate(steeringVector).T,steeringVector)
    steeringVector = steeringVector / weight

    return steeringVector


def frequency_spectrum(multi_channel_data,fs = 44100,num_channels = 16):
    frequencies = fftfreq(len(multi_channel_data[0]),1/fs)
    frequency_array = np.zeros((num_channels,len(frequencies)), dtype=np.complex)

    for i in range(num_channels):
        temp_freq = fft(multi_channel_data[i])
        frequency_array[i] = temp_freq

    max_freq = max(frequencies)

    return frequency_array, frequencies, max_freq


def phase_steering_vector(frequencies, mics_positions, az_angle, el_angle, num_mics = 16, c = 343):
    freq_len = len(frequencies)

    wave_vector = np.array(wave_unit_vector(az_angle, el_angle))
    direction = np.zeros(num_mics)
    for i in range(num_mics):
        direction[i] = wave_vector.dot(mics_positions[i,:])

    steeringVector = np.zeros((num_mics, freq_len), dtype=np.complex)
    for f in range(freq_len):
        t0 = direction / c
        steeringVector[:, f] = (np.exp(-1j * 2 * np.pi * frequencies[f] * t0))
    steeringVectorNormalised = normalise(steeringVector,freq_len)

    return steeringVectorNormalised


def get_spatial_correlation_matrix(frequency_spectrum,num_mics):
    freq_len = len(frequency_spectrum[0])
    correlation_matrix = np.zeros((num_mics,num_mics,freq_len),dtype=np.complex)

    for f in range(freq_len):
        correlation_matrix[:,:,f] = np.multiply.outer(np.conj(frequency_spectrum[:,f]).T,frequency_spectrum[:,f])

    return correlation_matrix


def get_beamformer_weights(steering_vector, correlation_matrix, frequency_spectrum, mic_num):
    freq_len = len(frequency_spectrum)
    weights = np.zeros((mic_num, freq_len), dtype=np.complex)
    for f in range(freq_len):
        cor_reshape = np.reshape(correlation_matrix[:,:,f],[mic_num,mic_num])
        cor_inv = np.linalg.pinv(cor_reshape)
        cor_inv = np.matrix(cor_inv)
        steering_h = np.matrix(steering_vector[:,f]).H
        denom_temp = np.matmul(np.conjugate(steering_vector[:,f]).T,cor_inv)
        denominator = np.matmul(denom_temp, steering_vector[:,f])
        denominator = np.reshape(denominator,[1,1])
        numerator = np.matmul(cor_inv, steering_vector[:,f])

        weights[:,f] = numerator / denominator
    return weights


def apply_weights(beamformer_weights,fft_signal,num_mics,frequencies):
    freq_len = len(frequencies)

    beamformed_signal = np.zeros((num_mics,freq_len),dtype=complex)
    for f in range(freq_len):
        beamformed_signal[:,f] = np.conj(beamformer_weights[:,f]) * fft_signal[:,f]

    return beamformed_signal


def get_spectrogram(multi_channel_time_data, fft_len, sample_shift):
    temp_channel_data = np.array(multi_channel_time_data)
    temp_channel_data = temp_channel_data / np.max(np.abs(temp_channel_data))
    num_channels, signal_len = temp_channel_data.shape

    number_of_frames = int((signal_len - fft_len) // sample_shift)

    spectrogram = np.zeros((num_channels, number_of_frames, int(fft_len // 2) + 1), dtype=np.complex64)
    start = int(0)
    end = int(fft_len)
    for i in range(number_of_frames):
        temp_data = fft(temp_channel_data[:,start:end], n=fft_len)[:, 0:int(fft_len//2)+1]
        start += sample_shift
        end += sample_shift
        spectrogram[:,i,:] = temp_data

    return spectrogram


def get_power_spectral_density_matrix(multi_channel_data, fft_len, sample_shift, frames_to_view = 15, start_point=0):
    temp_signal_data = np.array(multi_channel_data)
    num_mics, signal_len = temp_signal_data.shape

    power_spectral_density_matrix = np.zeros((num_mics,num_mics,fft_len), dtype = np.complex64)
    start = start_point
    end = start + fft_len
    frames_used = 0
    for i in range(frames_to_view):
        temp_signal_frequency = fft(temp_signal_data[:, start:end])
        for f in range(fft_len):
            power_spectral_density_matrix[:, :, f] += np.multiply.outer(temp_signal_frequency[:, f], np.conjugate(temp_signal_frequency[:, f]).T)

        start += sample_shift
        end += sample_shift
        frames_used += 1
        if end + sample_shift >= signal_len:
            break
    print("frames_used: " + str(frames_used))
    return power_spectral_density_matrix / frames_used


def get_beamforming_weights(correlation_matrix, steering_vector, fft_len):
    num_mics, _ = steering_vector.shape
    beamformer_weights = np.zeros((num_mics, int(fft_len / 2 + 1)), dtype=np.complex64)
    print("corelation: " + str(correlation_matrix.shape))
    for f in range(int(fft_len / 2 + 1)):
        inv_correlation_matrix = np.linalg.pinv(correlation_matrix[:, :, f])
        steering_vector_slice = steering_vector[:, f]

        numerator = np.matmul(inv_correlation_matrix, steering_vector_slice)
        denominator = np.matmul(np.conj(steering_vector_slice).T, inv_correlation_matrix)
        denominator = np.matmul(denominator, steering_vector_slice)
        beamformer_weights[:, f] = numerator / denominator

    return beamformer_weights


def get_steering_vector(fft_len, mics_positions, az_angle, el_angle, num_mics = 16, c = 343, fs=44100):
    steering_vector = np.zeros((num_mics, int(fft_len / 2 + 1)), dtype=np.complex64)

    freq_len = fft_len

    wave_vector = np.array(wave_unit_vector(az_angle, el_angle))
    direction = np.zeros(num_mics)
    for i in range(num_mics):
        direction[i] = wave_vector.dot(mics_positions[i, :])


    frequencies = fftfreq(fft_len, 1/fs)
    for f in range(int(freq_len / 2 + 1)):
        t0 = direction / c
        steering_vector[:, f] = (np.exp(-1j * 2 * np.pi * frequencies[f] * t0))
    steering_vector_normalised = normalise(steering_vector, freq_len)

    return steering_vector_normalised


def apply_beamforming_weights(beamforming_weights, complex_spectrogram):
    num_mics, num_frames, num_freqs = complex_spectrogram.shape
    weighted_signal = np.zeros((num_frames, num_freqs), dtype=np.complex64)
    for f in range(num_freqs):
        weighted_signal[:, f] = np.matmul(np.conj(beamforming_weights[:, f]).T, complex_spectrogram[:, :, f])

    return weighted_signal


def get_wav_signal(beamformed_signal, fft_len, sample_shift):
    num_frames, half_frequencies = beamformed_signal.shape
    window = signal.windows.hann(fft_len + 1, 'periodic')[: -1]

    temp_signal = np.zeros(fft_len, dtype=np.complex64)
    wav_signal = np.zeros((num_frames * (sample_shift+300)), dtype=np.complex64)
    start = 0
    end = start + fft_len
    for i in range(num_frames):
        spec = beamformed_signal[i, :]
        temp_signal[0:np.int(fft_len/2 + 1)] = spec.T
        temp_signal[np.int(fft_len/2 + 1):] = np.flip(np.conj(spec[1:np.int(fft_len/2)]))
        real_sig = ifft(temp_signal)

        wav_signal[start:end] = wav_signal[start:end] + real_sig * window.T

        start += sample_shift
        end += sample_shift

    return wav_signal[:end - sample_shift]

def get_covaviance_matrix(multi_channel_signal, K):
    num_mics, _ = multi_channel_signal.shape
    covariance_matrix = np.zeros((num_mics,num_mics),dtype=np.complex64)
    x_t = np.array(multi_channel_signal)
    for i in range(K):
        x_k = np.matrix(x_t[:, i])
        x_h = np.matrix(x_k.H)
        covariance_matrix += np.matmul(x_k, x_h)

    covariance_matrix = 1 / K * covariance_matrix

    return covariance_matrix

def alternate_weights(correlation_matrix, steering_vector, fft_len):
    num_mics, _ = steering_vector.shape
    beamformer_weights = np.zeros((num_mics, int(fft_len / 2 + 1)), dtype=np.complex64)
    print("corelation: " + str(correlation_matrix.shape))
    inv_correlation_matrix = np.linalg.pinv(correlation_matrix[:, :])
    for f in range(int(fft_len / 2 + 1)):

        steering_vector_slice = steering_vector[:, f]

        numerator = np.matmul(inv_correlation_matrix, steering_vector_slice)
        denominator = np.matmul(np.conj(steering_vector_slice).T, inv_correlation_matrix)
        denominator = np.matmul(denominator, steering_vector_slice)
        beamformer_weights[:, f] = numerator / denominator

    return beamformer_weights


def add_sixteen_channels_for_bullshit(channels):
    channelAve = []
    incr = 0
    channel_lengths = []
    for i in range(16):
        channel_lengths.append(len(channels[i]))
    smallest = min(channel_lengths)
    while incr != smallest:
        ave = 0
        for i in range(16):
            ave += (-1)**i * channels[i][incr]
        ave = ave / 16
        #channelAve.append(np.int16(round(ave, 0)))
        channelAve.append(np.cdouble(ave))
        incr += 1

    channelAve = np.array(channelAve)
    return channelAve


def delay_time_for_null_steering(mics_pos,az,el,c):
    az = az - 90
    el = el

    times = []
    wav = wave_unit_vector(az, el)
    for i in mics_pos:
        dist = i[0]*wav[0]+i[1]*wav[1]

        t = dist/c
        times.append(t)
    return times


def angle_from_times(multi_channel_data, mic1, mic2, mic_pos1, mic_pos2, mxlag, samples_to_use, fs=44100, c = 343):
    mic1 = mic1 - 1
    mic2 = mic2 - 1
    mic_separation = mic_pos2[0] - mic_pos1[0]
    cor = corelate_with_max(multi_channel_data[mic1,0:samples_to_use],multi_channel_data[mic2,0:samples_to_use], mxlag)
    index = np.where(cor[1] == max(cor[1]))
    sample_diff = cor[0][index[0][0]]
    straight_line_dist = int(sample_diff) * 1 / fs * c
    angle = np.arccos(straight_line_dist / mic_separation)
    angle = angle * 180 / np.pi
    #plt.figure()
    #plt.plot(cor[1])
    return angle


def get_null_steer_weights_x(fft_len, frequencies, null_ang1, null_ang2, null_ang3, el, el1, el2):

    null_weights = np.zeros((4,fft_len),dtype=complex)
    for i in range(fft_len):
        f = frequencies[i]
        z0 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.cos(null_ang1) * math.cos(el))
        z1 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.cos(null_ang2) * math.cos(el1))
        z2 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.cos(null_ang3) * math.cos(el2))
        null_weights[0,i] = -z0 * z1 * z2
        null_weights[1,i] = z0 * z1 + z0 * z2 + z1 * z2
        null_weights[2,i] = -z0 - z1 - z2
        null_weights[3,i] = 1

    return null_weights


def get_null_steer_weights_y(fft_len, frequencies, el, el1, el2):

    null_weights = np.zeros((4,fft_len),dtype=complex)
    for i in range(fft_len):
        f = frequencies[i]
        z0 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(el))
        z1 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(el1))
        z2 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(el2))
        null_weights[0,i] = -z0 * z1 * z2
        null_weights[1,i] = z0 * z1 + z0 * z2 + z1 * z2
        null_weights[2,i] = -z0 - z1 - z2
        null_weights[3,i] = 1

    return null_weights


def get_full_null_weights(weight_x, weight_y,fft_len):
    null_weights = np.zeros((4,4,fft_len),dtype=complex)

    for x in range(4):
        for y in range(4):
            for f in range(fft_len):
                null_weights[x, y, f] = weight_x[x, f] * weight_y[y, f]

    return null_weights


def get_null_steer_weights(fft_len, frequencies, null_ang1, null_ang2, null_ang3):
    null_weights = np.zeros((4,fft_len),dtype=complex)
    for i in range(fft_len):
        f = frequencies[i]
        z0 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(null_ang1))
        z1 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(null_ang2))
        z2 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * math.sin(null_ang3))
        null_weights[0,i] = -z0 * z1 * z2
        null_weights[1,i] = z0 * z1 + z0 * z2 + z1 * z2
        null_weights[2,i] = -z0 - z1 - z2
        null_weights[3,i] = 1

    return null_weights


def get_null_steer_weights_hopefully(fft_len, frequencies, null_ang1,null_num,d=0.042,c=343):
    null_weights = np.zeros((16,fft_len),dtype=complex)
    for f in range(fft_len):
        inv_lamda = np.float32(frequencies[f] / c)
        for i in range(4):
            for p in range(4):
                phase_shift = (2*np.pi*null_num/4) - (((2*np.pi*d)*inv_lamda)*math.cos(null_ang1))
                if frequencies[f] >= 0:
                    null_weights[4*i+p, f] = complex(np.exp(1j * p * phase_shift))
                else:
                    null_weights[4 * i + p, f] = complex(np.exp(-1j * p * phase_shift))

    return null_weights


def get_null_steer_weights_test(fft_len, frequencies, null_ang1, null_ang2, null_ang3,el,el1,el2):

    null_weights = np.zeros((4,fft_len),dtype=complex)
    for i in range(fft_len):
        f = frequencies[i]
        z0 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * (math.cos(null_ang1)+math.sin(el)))
        z1 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * (math.cos(null_ang2)+math.sin(el1)))
        z2 = np.exp(-1j * 2 * f / 343 * np.pi * 0.042 * (math.cos(null_ang3)+math.sin(el2)))
        null_weights[0,i] = -z0 * z1 * z2
        null_weights[1,i] = z0 * z1 + z0 * z2 + z1 * z2
        null_weights[2,i] = -z0 - z1 - z2
        null_weights[3,i] = 1

    return null_weights


def corelate_with_max(ch1, ch2, mxlag):
    a = np.correlate(ch1, ch2, mode='full')

    zero_index = int((len(a) + 1) / 2 - 1)
    limited_correlation = np.array(a[zero_index - mxlag: zero_index + mxlag + 1])
    shifts = np.arange(- mxlag, mxlag + 1, 1)

    return shifts, limited_correlation


def get_phase_difference(ch1,ch2,freq,fs):
    X1 = fft(ch1)
    X2 = fft(ch2)
    xf = fftfreq(len(ch1), 1 / fs)
    in1 = find_index(X1, xf, freq, 30, show=False)
    in2 = find_index(X2, xf, freq, 30, show=False)
    Y = complex(complex(X2[in2]) / complex(X1[in1]))
    phase_difference = np.angle(Y)

    return phase_difference


def get_angle_of_arrival(multi_channel, ch1,ch2, mic1, mic2,c,f,fs=44100):
    ch1 = ch1
    ch2 = ch2

    c1 = multi_channel[ch1]
    c2 = multi_channel[ch2]
    phase_diff = get_phase_difference(c1,c2,f,fs)
    pos_dif = np.abs(mic2[0]-mic1[0])
    limit = max_phase_allowed(f,c,pos_dif)

    if np.abs(phase_diff) < np.abs(limit):
        angle_of_arrival = np.arccos(((c/(2*np.pi*f))*(phase_diff/pos_dif)))*180/np.pi
    else:
        #debug("phase difference is "+str(phase_diff)+" while limit is "+str(limit))
        angle_of_arrival = None
    return angle_of_arrival, phase_diff


def get_angle_of_arrival_y(multi_channel, ch1,ch2, mic1, mic2,c,f,fs=44100):
    ch1 = ch1
    ch2 = ch2

    c1 = multi_channel[ch1]
    c2 = multi_channel[ch2]
    phase_diff = get_phase_difference(c1,c2,f,fs)
    pos_dif = np.abs(mic2[1]-mic1[1])
    limit = max_phase_allowed(f,c,pos_dif)

    if np.abs(phase_diff) < np.abs(limit):
        angle_of_arrival = np.arccos(((c/(2*np.pi*f))*(phase_diff/pos_dif)))*180/np.pi
    else:
        # debug("phase difference is "+str(phase_diff)+" while limit is "+str(limit))
        angle_of_arrival = None
    return angle_of_arrival, phase_diff


def get_phase(complex_num):
    complex_num = complex(complex_num)
    phase = np.arctan(np.imag(complex_num)/np.real(complex_num))
    return phase


def rattle_off_phase_diffs(multi_channel_signal,row_by_channel_nums,freq, mic_positions, fs=44100,axis="Row",c=343):
    row_by_channel_nums = np.array(row_by_channel_nums) -1
    angle_of_arrival=0
    #print("Single spacing")
    singles = []
    doubles = []
    triples = []
    triples_angle = []
    if axis == "Row":
        for i in range(3):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i+1]],freq,fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival(multi_channel_signal, row_by_channel_nums[i],row_by_channel_nums[i+1],
                                       mic_positions[i], mic_positions[i + 1], 343, freq, fs)
            if angle and angle != 180:
                singles.append(angle)

            #print(angle)
        #print("Double spacing")
        for i in range(2):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i + 2]], freq, fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival(multi_channel_signal, row_by_channel_nums[i], row_by_channel_nums[i + 2],
                                       mic_positions[i], mic_positions[i + 2], 343, freq, fs)
            if angle and angle != 180:
                doubles.append(phase)
            #print(angle)
        #print("Triple spacing")
        for i in range(1):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i + 3]], freq, fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival(multi_channel_signal, row_by_channel_nums[i], row_by_channel_nums[i + 3],
                                       mic_positions[i], mic_positions[i + 3], 343, freq, fs)
            if angle and angle != 180:
                triples_angle.append(angle)
                triples.append(phase)
            else:
                triples.append(False)
            #print(angle)
        baselines = determine_baseline(freq)
        if triples[0]:
            if baselines[2] == True:
                angle_of_arrival = triples_angle[0]
                ambiguities = False
            elif baselines[2] == False and baselines[0] == True:
                if singles:
                    angle_of_arrival = interferometry_resolution(singles, triples[0], freq, c, d=0.042)

                ambiguities = False
            else:
                angle_of_arrival = interferometry_resolution(singles, triples[0], freq, c, d=0.042, ambiguities=True)
                ambiguities = True

    else:
        # print("Single spacing")
        singles = []
        doubles = []
        triples = []
        triples_angle = []
        for i in range(3):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i+1]],freq,fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival_y(multi_channel_signal, row_by_channel_nums[i],
                                                row_by_channel_nums[i + 1],
                                                mic_positions[i], mic_positions[i + 1], 343, freq, fs)
            if angle and angle != 180:
                singles.append(angle)
            # print(angle)
        # print("Double spacing")
        for i in range(2):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i + 2]], freq, fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival_y(multi_channel_signal, row_by_channel_nums[i],
                                                row_by_channel_nums[i + 2],
                                                mic_positions[i], mic_positions[i + 2], 343, freq, fs)
            if angle and angle != 180:
                doubles.append(phase)
            # print(angle)
        # print("Triple spacing")
        for i in range(1):
            """phase = get_phase_difference(multi_channel_signal[row_by_channel_nums[i]],
                                         multi_channel_signal[row_by_channel_nums[i + 3]], freq, fs)
            print(phase)"""
            angle, phase = get_angle_of_arrival_y(multi_channel_signal, row_by_channel_nums[i],
                                                row_by_channel_nums[i + 3],
                                                mic_positions[i], mic_positions[i + 3], 343, freq, fs)
            if angle and angle != 180:
                triples_angle.append(angle)
                triples.append(phase)
            else:
                triples.append(False)
            # print(angle)

        baselines = determine_baseline(freq)
        if triples[0]:
            if baselines[2] == True:
                angle_of_arrival = triples_angle[0]
                ambiguities = False
            elif baselines[2] == False and baselines[0] == True:
                angle_of_arrival = interferometry_resolution(singles, triples[0], freq, c, d=0.042,angle="el")

                ambiguities = False
            else:
                angle_of_arrival = interferometry_resolution(singles, triples[0], freq, c, d=0.042, ambiguities=True,angle="el")
                ambiguities = True

    """print("singles = "+str(singles))
    print("doubles = "+str(np.array(doubles)*180/np.pi))
    print("triples = "+str(np.array(triples)))"""
    """
    print()
    print(np.array(doubles) * 180 / np.pi)
    print()
    print(np.array(triples) * 180 / np.pi)
    print()"""
    array_weighting = get_array_weighting(singles)
    return angle_of_arrival, array_weighting

def determine_baseline(f, d=0.042, c=343):
    """
    returns array of three elements all boolean values. Each describes whether the baseline for a spacing is long or not.
    the array position describes the number of spacings (1-3 in this case)
    """
    lamda = c/f
    baselines = [False,False,False]
    for i in range(3):

        if (d*(i+1)) <= (lamda/2):
            baselines[i] = True

    return baselines


def max_phase_allowed(f,c,d):
    max_allowed = 1/((c/f)/(2*np.pi*d))
    return max_allowed


def interferometry_resolution(singles_angles, phase2, f, c, d,ambiguities=False,angle="az" ):
    angle_decision = None
    lamda = c/f
    wraps = int((3*d)/(lamda/2))
    lim = max_phase_allowed(f,c,3*d)
    angle_options = []
    if angle == "az":
        iterate = np.arange(-(wraps),(wraps)+1,1)
        for i in iterate:
            if np.abs(phase2 + (2*np.pi * i)) <= lim:
                val = np.arccos(((phase2 + (2 * np.pi * i)) * (c / f)) / (2 * np.pi * 3 * d)) * 180 / np.pi
                angle_options.append(val)
            if np.abs(phase2 + 2 * (np.pi * -i)) <= lim and i != 0:
                val = np.arccos(((phase2 + (2 * np.pi * -i)) * (c / f)) / (2 * np.pi * 3 * d)) * 180 / np.pi
                angle_options.append(val)
        if ambiguities == False:
            likely_angles = []
            for i in singles_angles:
                likely_angle = i - angle_options[0]
                for angle_option in angle_options:
                    if np.abs(i-angle_option) < np.abs(i-likely_angle):
                        likely_angle = angle_option
                likely_angles.append(likely_angle)
            if likely_angles:
                angle_decision = mode(np.array(likely_angles))
        else:
            angle_decision = angle_options
    else:
        iterate = np.arange(-(wraps), (wraps) + 1, 1)
        for i in iterate:
            if np.abs(phase2 + (2 * np.pi * i)) <= lim:
                val = np.arccos(((phase2 + (2 * np.pi * i)) * (c / f)) / (2 * np.pi * 3 * d)) * 180 / np.pi
                angle_options.append(val)
            if np.abs(phase2 + 2 * (np.pi * -i)) <= lim and i != 0:
                val = np.arccos(((phase2 + (2 * np.pi * -i)) * (c / f)) / (2 * np.pi * 3 * d)) * 180 / np.pi
                angle_options.append(val)
        if ambiguities == False:
            likely_angles = []
            for i in singles_angles:
                likely_angle = i - angle_options[0]
                for angle_option in angle_options:
                    if np.abs(i - angle_option) < np.abs(i - likely_angle):
                        likely_angle = angle_option
                likely_angles.append(likely_angle)
            if likely_angles:
                angle_decision = mode(np.array(likely_angles))
        else:
            angle_decision = angle_options
    if angle_decision:
        return angle_decision


def plot_fourier(signal,fs = 44100):
    X = fft(signal)
    xf = fftfreq(len(signal), 1/fs)
    plt.figure()
    plt.plot(xf, np.abs(X))
    plt.show()


def get_channel_data(filename):
    wav = wave.open(filename)
    i = wav.getnframes()

    nch = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())
    wav.close()

    inChannels = []
    #print("Extracting Data")
    for channel in range(16):
        #print(channel)
        typ = {1: np.int8, 2: np.int16, 4: np.int32}.get(depth)
        if not typ:
            raise ValueError("sample width {} not supported".format(depth))
        if channel >= nch:
            raise ValueError("cannot extract channel {} out of {}".format(+1, nch))
        #print("Extracting channel {} out of {} channels, {}-bit depth".format(channel + 1, nch, depth * 8))
        data = np.frombuffer(sdata, dtype=typ)
        ch_data = data[channel::nch]
        inChannels.append(ch_data)

    inChannels = np.array(inChannels, dtype=np.float32)

    return inChannels


def debug(message=""):
    if message:
        print("debug: "+str(message))
    else:
        print("debug")


def remove_outliers(array):
    array = np.array(array)
    mean = sum(array)/len(array)
    tot = float(0)
    for i in array:
        tot += (i-mean)**2
    std_dev = np.sqrt(float(tot/len(array)))
    fixed_data = []

    for i in array:
        if float(i) > mean + 2*std_dev or float(i) < mean - 2*std_dev:
            print("outlier found in "+ str(array))
            print("outlier is "+str(i))

        else:
            fixed_data.append(i)

    fixed_data = np.array(fixed_data)

    return fixed_data


def get_array_weighting(array):
    array = np.array(array)
    var = np.var(array)
    if len(array) == 4 and var < 30:
        weighting = 3
    else:
        weighting = 1
    return weighting


def scale_for_wav_saving(signal):
    """Scales the signal to the required saving format (in this case int32)"""
    highest_val = max(np.abs(signal))
    if highest_val < 60000000:
        scaled_signal = signal / highest_val * 1000000000
    else:
        scaled_signal = signal / highest_val * 2000000000

    return np.int32(real(scaled_signal))


def bit_conversion_24_32(byte_array):
    ar_length = int(len(byte_array))
    num_array = np.zeros(int(ar_length/3),dtype=np.int32)
    for i in range(int(ar_length/3)):
        num = np.int32(int.from_bytes(byte_array[3*i:3*i+3],"little",signed=True))
        num_array[i] = num
    converted_byte_array = num_array.tobytes()

    return converted_byte_array


def get_data(sdata):
    nch = 16
    inChannels = []
    #print("Extracting Data")
    for channel in range(16):
        # print(channel)
        typ = np.int32

        if channel >= nch:
            raise ValueError("cannot extract channel {} out of {}".format(+1, nch))
        # print("Extracting channel {} out of {} channels, {}-bit depth".format(channel + 1, nch, depth * 8))
        data = np.frombuffer(sdata, dtype=typ)
        ch_data = data[channel::nch]
        inChannels.append(ch_data)

    inChannels = np.array(inChannels, dtype=np.float32)

    return inChannels

def record(filename, seconds):
    sample_format = pyaudio.paInt24  # 32 bits per sample
    channels = 16
    fs = 44100  # Record at 44100 samples per second
    chunk = fs * seconds  # Record in chunks of 1024 samples

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    #print(p.get_device_info_by_index(17))
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    start_time = time.time()
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("--- %s seconds ---" % (time.time() - start_time))
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    print('Finished recording')

    data_32bit = bit_conversion_24_32(frames[0])

    sample_format = pyaudio.paInt32
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(data_32bit)
    wf.close()

    return


def apply_x_nulls(weights, ordered_signal):
    fourier = np.zeros((16, len(weights[0])), dtype=complex)
    weighted_signal = np.zeros((16, len(weights[0])), dtype=complex)
    freq_length = len(weights[0])
    xf = fftfreq(freq_length,1/44100)
    for i in range(16):
        fourier[i, :] = fft(ordered_signal[i])
        for f in range(freq_length):
            weighted_signal[i, f] = fourier[i, f] * np.conj(weights[i, f])

    final_signal = add_sixteen_channels(weighted_signal)
    result = ifft(final_signal)
    return result


def get_steer_weights(fft_len, frequencies, steer_angle, d=0.042, c=343, axis='x'):
    steer_weights = np.zeros((16, fft_len),dtype=complex)
    for f in range(fft_len):
        inv_lamda = np.float32(frequencies[f] / c)
        for i in range(4):
            for p in range(4):
                if axis == 'x':
                    phase_shift = - (((2*np.pi*d)*inv_lamda)*math.cos(steer_angle))
                else:
                    phase_shift = - (((2*np.pi*d)*inv_lamda)*math.cos(steer_angle))

                if frequencies[f] >= 0:
                    steer_weights[4*i+p, f] = complex(np.exp(1j * p * phase_shift))
                else:
                    steer_weights[4 * i + p, f] = complex(np.exp(-1j * p * phase_shift))

    return steer_weights


"""def get_beam_and_null_weights(fft_len, frequencies, steer_angle, null_angle, d=0.042, c=343):
    optimal_weights = np.zeros((4, fft_len), dtype=complex)
    C = np.zeros((4, fft_len), dtype=complex)
    Wd = np.zeros((4, fft_len), dtype=complex)

    optimal_weights = np.matrix(optimal_weights)
    C = np.matrix(C)
    Wd = np.matrix(Wd)

    for f in range(fft_len):
        inv_lamda = np.float32((frequencies[f]) / c)
        for i in range(4):

            phase_shift = 2 * np.pi * d * inv_lamda * math.sin(steer_angle)
            phase_null = 2 * np.pi * inv_lamda * d * math.cos(null_angle)
            if int(f) == 1500:
                print(phase_shift)
                print(phase_null)
            if frequencies[f] >= 0:
                Wd[i, f] = complex(np.exp(-1j * (-1.5 + i) * phase_shift))
                C[i, f] = complex(np.exp(-1j * (-1.5 + i) * phase_null))
            else:
                Wd[i, f] = complex(np.exp(-1j * (-1.5 + i) * phase_shift))
                C[i, f] = complex(np.exp(-1j * (-1.5 + i) * phase_null))

        inv_C = np.matmul(C[:, f].H, C[:, f]).I
        C_combo = C[:, f] * inv_C
        C_combo_2 = np.matmul(C_combo, C[:, f].H)
        adding_w = Wd[:, f].H - np.matmul(Wd[:, f].H, C_combo_2)
        adding_w = Wd[:, f] - np.matmul(C_combo_2, Wd[:, f])

        if int(f) == 2000 or int(f) == -2000:
            print(f)
            print(C[:,f])
            print(C_combo_2)
            print(np.matmul(Wd[:, f].H, C_combo_2))
            print(Wd[:,f])

        optimal_weights[:, f] = adding_w

    return np.array(optimal_weights)
"""


def get_beam_and_null_weights(fft_len, frequencies, steer_angle, null_angle, d=0.042, c=343):
    optimal_weights = np.zeros((4, fft_len), dtype=complex)
    C = np.zeros((4, fft_len), dtype=complex)
    Wd = np.zeros((4, fft_len), dtype=complex)

    optimal_weights = np.matrix(optimal_weights)
    C = np.matrix(C)
    Wd = np.matrix(Wd)

    for f in range(fft_len):
        inv_lamda = np.float32((frequencies[f]) / c)
        for i in range(4):
            phase_shift = 2 * np.pi * d * inv_lamda * math.cos(steer_angle)
            phase_null = 2 * np.pi * inv_lamda * d * math.cos(null_angle)

            if frequencies[f] >= 0:
                Wd[i, f] = complex(np.exp(-1j * i * phase_shift))
                C[i, f] = complex(np.exp(-1j * i * phase_null))
            else:
                Wd[i, f] = complex(np.exp(-1j * i * phase_shift))
                C[i, f] = complex(np.exp(-1j * i * phase_null))

        inv_C = np.matmul(C[:, f].H, C[:, f]).I
        C_combo = C[:, f] * inv_C
        C_combo_2 = np.matmul(C_combo, C[:, f].H)
        adding_w = Wd[:, f].H - np.matmul(Wd[:, f].H, C_combo_2)

        optimal_weights[:, f] = adding_w.H

    return np.array(optimal_weights)


def plot_psd(signal):
    f, Pxx = welch(signal, fs=44100, nperseg=len(signal) // 2, return_onesided=False)
    plt.figure()
    plt.plot(f,Pxx)
    plt.show()


def steer_for_resolution(channel_data, steer_angle, tone_freq):
    fs = 44100
    ang = steer_angle * np.pi / 180

    mic1 = [0.021, -0.063, 0]
    mic2 = [0.063, -0.063, 0]
    mic3 = [0.021, -0.021, 0]
    mic4 = [0.063, -0.021, 0]
    mic5 = [0.021, 0.021, 0]
    mic6 = [0.063, 0.021, 0]
    mic7 = [0.021, 0.063, 0]
    mic8 = [0.063, 0.063, 0]
    mic9 = [-0.063, 0.063, 0]
    mic10 = [-0.021, 0.063, 0]
    mic11 = [-0.063, 0.021, 0]
    mic12 = [-0.021, 0.021, 0]
    mic13 = [-0.063, -0.021, 0]
    mic14 = [-0.021, -0.021, 0]
    mic15 = [-0.063, -0.063, 0]
    mic16 = [-0.021, -0.063, 0]

    micsR1 = [mic8, mic7, mic10, mic9]
    micsR2 = [mic6, mic5, mic12, mic11]
    micsR3 = [mic4, mic3, mic14, mic13]
    micsR4 = [mic2, mic1, mic16, mic15]

    micR1_nums = [8, 7, 10, 9]
    micR2_nums = [6, 5, 12, 11]
    micR3_nums = [4, 3, 14, 13]
    micR4_nums = [2, 1, 16, 15]

    micCols_num = [micR1_nums, micR2_nums, micR3_nums, micR4_nums]

    inChannels = channel_data
    xf = fftfreq(len(inChannels[1]), 1 / fs)
    weights = get_steer_weights(len(xf), xf, ang)

    temp = np.zeros((16, len(inChannels[1])), dtype=complex)
    for i in range(4):
        for p in range(4):
            temp[4 * i + p, :] = inChannels[micCols_num[i][p] - 1, :]

    temp1 = np.zeros((16, len(inChannels[0])), dtype=complex)

    for i in range(4):
        for p in range(4):
            X = fft(temp[4 * i + p, :])
            xf = fftfreq(len(temp[i]), 1 / fs)
            weighted = np.zeros(len(temp[i]), dtype=complex)
            for f in range(len(xf)):
                weighted[f] = X[f] * np.conj(weights[p, f])
            shift = ifft(weighted)

            temp1[4 * i + p] = shift

    final = add_sixteen_channels(temp1)
    s = snr(final, tone_freq)
    return s

def make_array_of_likely_angles(array_of_arrays):
    max_len = 0
    array_length = len(array_of_arrays)
    reference_id = 0
    temp_list = []
    for i in array_of_arrays:
        temp_list.append(remove_duplicates(i))
    new_array_of_arrays = np.array(temp_list,dtype=object)
    for i in range(array_length):
        if len(new_array_of_arrays[i]) > max_len:
            max_len = len(new_array_of_arrays[i])
            reference_id = i
    likely_angles = np.zeros(max_len)
    for i in range(max_len):
        total = new_array_of_arrays[reference_id][i]
        count = 1
        for p in range(array_length):
            if p != reference_id:
                for elem in new_array_of_arrays[p]:
                    if elem-5 < new_array_of_arrays[reference_id][i] < elem+5:

                        total += elem
                        count += 1
        likely_angles[i] = total / count
    return likely_angles


def find_snrs(channel_data, array_of_angles, tone_freq):
    snrs = np.zeros(len(array_of_angles))
    for i in range(len(array_of_angles)):
        temp_snr = steer_for_resolution(channel_data, array_of_angles[i], tone_freq)
        snrs[i] = temp_snr

    return snrs


def get_angle_from_snrs(snrs, likely_angles):
    max_snr = max(snrs)
    id = 0
    for i in range(len(snrs)):
        if snrs[i] == max_snr:
            id = i
    angle = likely_angles[id]

    return angle


def remove_duplicates(array):
    temp_list = []
    for i in array:
        count = 0
        for p in temp_list:
            if np.round(i,3) == np.round(p,3):
                count += 1
                break
        if count == 0:
            temp_list.append(i)

    duplicates_removed =(temp_list)
    return duplicates_removed


def get_null_weights(fft_len, frequencies, null_ang1,null_num,d=0.042,c=343):
    null_weights = np.zeros((16,fft_len),dtype=complex)
    for f in range(fft_len):
        inv_lamda = np.float32(frequencies[f] / c)
        for i in range(4):
            for p in range(4):
                phase_shift = (2*np.pi*null_num/4) - (((2*np.pi*d)*inv_lamda)*math.cos(null_ang1))
                if frequencies[f] >= 0:
                    null_weights[4*i+p, f] = complex(np.exp(1j * p * phase_shift))
                else:
                    null_weights[4 * i + p, f] = 1

    return null_weights


def get_null_weights_y(fft_len, frequencies, null_ang1,null_num,d=0.042,c=343):
    null_weights = np.zeros((16,fft_len),dtype=complex)
    for f in range(fft_len):
        inv_lamda = np.float32(frequencies[f] / c)
        for i in range(4):
            for p in range(4):
                phase_shift = (2*np.pi*null_num/4) - (((2*np.pi*d)*inv_lamda)*math.cos(null_ang1))

                if frequencies[f] >= 0:
                    null_weights[4*i+p, f] = complex(np.exp(-1j * p * phase_shift))
                else:
                    null_weights[4 * i + p, f] = complex(np.exp(1j * p * phase_shift))

    return null_weights


def apply_weighting(weights, signal, fs = 44100):
    sig_len = len(weights)
    weighted_signal = np.zeros(sig_len,dtype=complex)
    end_index = int(sig_len/2)
    xf = fftfreq(sig_len, 1 / fs)

    for i in range(end_index):
        weighted_signal[i] = np.conj(weights[i]) * signal[i]

    weighted_signal[end_index+1:] = np.flip(np.conj(weighted_signal[1:end_index]))
    return weighted_signal


def plot_data(signal):
    plt.figure()
    plt.plot(signal)
    plt.show()



def get_null_weights_for_row(fft_len, frequencies, null_ang1,null_num,d=0.042,c=343):
    null_weights = np.zeros((4,fft_len),dtype=complex)
    for f in range(fft_len):
        inv_lamda = np.float32(frequencies[f] / c)

        for p in range(4):
            phase_shift = (2*np.pi*null_num/4) - (((2*np.pi*d)*inv_lamda)*math.cos(null_ang1))
            if frequencies[f] >= 0:
                null_weights[p, f] = complex(np.exp(1j * p * phase_shift))
            else:
                null_weights[p, f] = 1

    return null_weights
