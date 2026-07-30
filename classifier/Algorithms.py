import numpy as np
import math
import statistics
#import own_detectors
import scipy.signal as signal
#import conf

HZ = 360

ALG2_LEVEL_WIDTH = 0.07
ALG2_A_THRES = 7

ALG3_LEVEL_WIDTH = 0.09
ALG3_K = 4
ALG3_W = 9
ALG3_RISE_AFTER_FALL = 0
ALG3_RISE_AFTER_RISE = 1
ALG3_FALL_AFTER_FALL = 2
ALG3_FALL_AFTER_RISE = 3
ALG3_COEFF1 = 0.25
ALG3_COEFF2 = 0.25
ALG3_COEFF3 = 0.125



def alg1_spanish(x): #, return_preprocessed_signal):
    Nd = 7
    N = 8
    rr_min = 72 # 200ms, (360 * 0.200)
    qrs_int = 21    # 60ms, (360 * 0.060) TODO: check that value, it is 21.6. Use 21 or 22?
    state1_length = rr_min + qrs_int
    y0 = []
    y1 = []
    y = []
    state = 1
    counter = 0
    max_peak_y = 0.0
    max_peak_x = 0
    found_peaks = []
    found_peaks_pos = []
    threshold_amplitude = 0.0
    th = None
    for n in range(0, len(x)):
        n_sub_Nd = max(n - Nd, 0)
        y0.append(x[n] - x[n_sub_Nd])

        sum_y0 = 0.0
        for k in range(0, N):
            n_sub_k = max(n - k, 0)
            sum_y0 += y0[n_sub_k]
        y1.append(sum_y0 / (N - 1))

        y.append(y1[n] * y1[n])

        if 1 == state:
            if y[n] >= max_peak_y:
                max_peak_y = y[n]
                max_peak_x = n
            # at the end of state1:
            counter += 1
            if counter == state1_length:
                # add peak
                found_peaks.append(max_peak_y)
                found_peaks_pos.append(max_peak_x)
                # update amplitude
                threshold_amplitude = sum(found_peaks) / len(found_peaks)
                # change state
                counter = 0
                state = 2
                max_peak_y = 0.0
        elif 2 == state:
            if n - max_peak_x >= rr_min:
                state = 3
        elif 3 == state:
            if th is None:
                th = threshold_amplitude
            else:
                th = th * SPANISH_TH_FACTOR
            if y[n] > th:
                th = None
                state = 1
    #if return_preprocessed_signal:
     #   return found_peaks_pos, y
    return found_peaks_pos


def alg1_get_initial_threshold(x, Nd, N):
    y0 = []
    y1 = []
    y = []
    for n in range(0, 3 * HZ):
        n_sub_Nd = max(n - Nd, 0)
        y0.append(x[n] - x[n_sub_Nd])

        sum_y0 = 0.0
        for k in range(0, N):
            n_sub_k = max(n - k, 0)
            sum_y0 += y0[n_sub_k]
        y1.append(sum_y0 / (N - 1))

        y.append(y1[n] * y1[n])
    return max(y)


def alg3_get_thresholds(val):
    lower_thres = 0.0
    upper_thres = ALG3_K * ALG3_LEVEL_WIDTH
    while val < lower_thres:
        lower_thres -= ALG3_LEVEL_WIDTH
        upper_thres -= ALG3_LEVEL_WIDTH
    while val > upper_thres:
        lower_thres += ALG3_LEVEL_WIDTH
        upper_thres += ALG3_LEVEL_WIDTH
    return lower_thres, upper_thres


# TODO: refactor
def alg3_generate_events(x):
    lower_thres, upper_thres = alg3_get_thresholds(x[0])
    # print('alg3 generating events: first sample value: ' + str(x[0]) + '\tlower_thres: ' + str(lower_thres) + '\tupper_thres:' + str(upper_thres))
    # print('alg3 generating events: first 0.100s (36 samples) values: ' + str(x[:36]))
    events = []
    x_len = len(x)
    for i in range(0, x_len):
        if x[i] < lower_thres:
            if 0 == len(events):
                events.append((ALG3_FALL_AFTER_FALL, i))
            else:
                if ALG3_RISE_AFTER_FALL == events[-1][0] or ALG3_RISE_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_FALL_AFTER_RISE, i))
                else:
                    events.append((ALG3_FALL_AFTER_FALL, i))
            lower_thres -= ALG3_LEVEL_WIDTH
            upper_thres -= ALG3_LEVEL_WIDTH
        elif x[i] > upper_thres:
            if 0 == len(events):
                events.append((ALG3_RISE_AFTER_RISE, i))
            else:
                if ALG3_FALL_AFTER_FALL == events[-1][0] or ALG3_FALL_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_RISE_AFTER_FALL, i)) # 0 - RISE after FALL
                else:
                    events.append((ALG3_RISE_AFTER_RISE, i))
            lower_thres += ALG3_LEVEL_WIDTH
            upper_thres += ALG3_LEVEL_WIDTH
    # print(events)
    return events


def is_peak(event_type):
    return ALG3_FALL_AFTER_RISE == event_type or ALG3_RISE_AFTER_FALL == event_type


def get_dur(events, p):
    start_idx = max(p - math.ceil(ALG3_W / 2.0), 0)
    end_idx = min(p + math.ceil(ALG3_W / 2.0) - ALG3_K + ALG3_W % 2, len(events) - 1)
    return int(events[end_idx][1] - events[start_idx][1])


# TODO: to check
def alg3_iranian(x):
    SP = None
    NP = None
    events = alg3_generate_events(x)
    # print(events[-100:])
    events_len = len(events)
    peaks = []
    pob = 300
    TH2 = 0.5 * pob
    for i in range(0, events_len):
        if is_peak(events[i][0]):
            dur = get_dur(events, i)
            if SP is None:
                SP = dur
                NP = dur
                TH1 = (SP + ALG3_COEFF2 * (NP - SP)) + 1
            # print('Alg3: sample: ' + str(events[i][1]) + '\tdur: ' + str(dur) + '\tTH1: ' + str(TH1))
            if dur <= TH1:
                if len(peaks) > 0:
                    if TH2 < abs(events[i][1] - peaks[-1]):
                        SP = SP - ALG3_COEFF1 * (SP - dur)
                        pob = min(pob - ALG3_COEFF3 * (pob - abs(events[i][1] - peaks[-1])), 360)
                        TH2 = 0.5 * pob
                        peaks.append(events[i][1])
                        TH1 = SP + ALG3_COEFF2 * (NP - SP)
                else:
                    SP = SP - ALG3_COEFF1 * (SP - dur)
                    peaks.append(events[i][1])
                    TH1 = 1 + (SP + ALG3_COEFF2 * (NP - SP))
            else:
                NP = NP - ALG3_COEFF1 * (NP - dur)
    # print('Tape evaluated by alg3 iranian')
    return peaks


def alg3_get_thresholds_operations(val, opers):
    lower_thres = 0.0
    opers['MUL'] += 1
    upper_thres = ALG3_K * ALG3_LEVEL_WIDTH
    while val < lower_thres:
        opers['SUB'] += 2
        lower_thres -= ALG3_LEVEL_WIDTH
        upper_thres -= ALG3_LEVEL_WIDTH
    while val > upper_thres:
        opers['ADD'] += 2
        lower_thres += ALG3_LEVEL_WIDTH
        upper_thres += ALG3_LEVEL_WIDTH
    return lower_thres, upper_thres


def alg3_generate_events_operations(x, opers):
    lower_thres, upper_thres = alg3_get_thresholds_operations(x[0], opers)
    # print('alg3 generating events: first sample value: ' + str(x[0]) + '\tlower_thres: ' + str(lower_thres) + '\tupper_thres:' + str(upper_thres))
    # print('alg3 generating events: first 0.100s (36 samples) values: ' + str(x[:36]))
    events = []
    x_len = len(x)
    for i in range(0, x_len):
        opers['COMP'] += 1
        if x[i] < lower_thres:
            opers['COMP'] += 1
            if 0 == len(events):
                events.append((ALG3_FALL_AFTER_FALL, i))
            else:
                opers['COMP'] += 2
                if ALG3_RISE_AFTER_FALL == events[-1][0] or ALG3_RISE_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_FALL_AFTER_RISE, i))
                else:
                    events.append((ALG3_FALL_AFTER_FALL, i))
            opers['SUB'] += 2
            lower_thres -= ALG3_LEVEL_WIDTH
            upper_thres -= ALG3_LEVEL_WIDTH
        elif x[i] > upper_thres:
            opers['COMP'] += 2
            if 0 == len(events):
                events.append((ALG3_RISE_AFTER_RISE, i))
            else:
                opers['COMP'] += 2
                if ALG3_FALL_AFTER_FALL == events[-1][0] or ALG3_FALL_AFTER_RISE == events[-1][0]:
                    events.append((ALG3_RISE_AFTER_FALL, i)) # 0 - RISE after FALL
                else:
                    events.append((ALG3_RISE_AFTER_RISE, i))
            opers['ADD'] += 2
            lower_thres += ALG3_LEVEL_WIDTH
            upper_thres += ALG3_LEVEL_WIDTH
    # print(events)
    return events


def get_dur_operations(events, p, opers):
    opers['SUB'] += 1
    opers['DIV'] += 1
    start_idx = max(p - math.ceil(ALG3_W / 2.0), 0)
    opers['ADD'] += 2
    opers['DIV'] += 1
    opers['SUB'] += 2
    end_idx = min(p + math.ceil(ALG3_W / 2.0) - ALG3_K + ALG3_W % 2, len(events) - 1)
    opers['SUB'] += 1
    return int(events[end_idx][1] - events[start_idx][1])


#TODO: subfunctions
def alg3_iranian_operations(x):
    opers = {'ADD': 0, 'SUB': 0, 'MUL': 0, 'DIV': 0, 'EXP': 0, 'COMP': 0, 'ABS': 0}
    SP = None
    NP = None
    events = alg3_generate_events_operations(x, opers)
    # print(events[-100:])
    events_len = len(events)
    peaks = []
    pob = 300
    opers['MUL'] += 1
    TH2 = 0.5 * pob
    for i in range(0, events_len):
        opers['COMP'] += 1
        if is_peak(events[i][0]):
            dur = get_dur_operations(events, i, opers)
            opers['COMP'] += 1
            if SP is None:
                SP = dur
                NP = dur
                opers['MUL'] += 1
                opers['ADD'] += 2
                opers['SUB'] += 1
                TH1 = (SP + ALG3_COEFF2 * (NP - SP)) + 1
            # print('Alg3: sample: ' + str(events[i][1]) + '\tdur: ' + str(dur) + '\tTH1: ' + str(TH1))
            opers['COMP'] += 1
            if dur <= TH1:
                opers['COMP'] += 1
                if len(peaks) > 0:
                    opers['COMP'] += 1
                    opers['SUB'] += 1
                    if TH2 < abs(events[i][1] - peaks[-1]):
                        opers['MUL'] += 1
                        opers['SUB'] += 2
                        SP = SP - ALG3_COEFF1 * (SP - dur)
                        opers['SUB'] += 3
                        opers['MUL'] += 1
                        pob = min(pob - ALG3_COEFF3 * (pob - abs(events[i][1] - peaks[-1])), 360)
                        opers['MUL'] += 1
                        TH2 = 0.5 * pob
                        peaks.append(events[i][1])
                        opers['MUL'] += 1
                        opers['SUB'] += 1
                        opers['ADD'] += 1
                        TH1 = SP + ALG3_COEFF2 * (NP - SP)
                else:
                    opers['MUL'] += 1
                    opers['SUB'] += 2
                    SP = SP - ALG3_COEFF1 * (SP - dur)
                    peaks.append(events[i][1])
                    opers['MUL'] += 1
                    opers['SUB'] += 1
                    opers['ADD'] += 2
                    TH1 = 1 + (SP + ALG3_COEFF2 * (NP - SP))
            else:
                opers['SUB'] += 2
                opers['MUL'] += 1
                NP = NP - ALG3_COEFF1 * (NP - dur)
    # print('Tape evaluated by alg3 iranian')
    return peaks, opers



def alg4_polish(x):
    ALPHA = 0.46
    GAMMA = 0.97
    samples_num_short_avg = 20
    samples_num_long_avg = 100
    samples_num_window = 72
    max_diff_arg, max_diff_val = alg4_find_first_peak(x)
    found_peaks = [max_diff_arg]
    threshold = ALPHA * max_diff_val
    short_avg_sum = 0.0
    long_avg_sum = 0.0
    search_samples_left = 0
    max_x = 0
    max_abs_y = 0.0
    refractory_window_end = found_peaks[0] + samples_num_window
    is_inside_refractory_window = True
    is_inside_searching_window = False
    x_len = len(x)
    max_new = 0.0
    for i in range(0, x_len):
        short_avg_sum += x[i]
        long_avg_sum += x[i]
        if i >= samples_num_short_avg:
            short_avg_sum -= x[i - samples_num_short_avg]
        if i >= samples_num_long_avg:
            long_avg_sum -= x[i - samples_num_long_avg]
        if i < samples_num_short_avg - 1:
            continue

        abs_diff_short = abs(short_avg_sum / samples_num_short_avg - x[i]) #TODO: czy tu na pewno ten short diff
        if is_inside_refractory_window:
            if i == refractory_window_end:
                is_inside_refractory_window = False
            else:
                continue
        if abs_diff_short >= threshold:
            if not is_inside_searching_window:
                is_inside_searching_window = True
                search_samples_left = samples_num_window
        if is_inside_searching_window:
            if abs(long_avg_sum / samples_num_long_avg - x[i]) > max_abs_y:
                max_x = i
                max_abs_y = abs(long_avg_sum / samples_num_long_avg - x[i])
                max_abs_short = abs_diff_short
            if abs(short_avg_sum / samples_num_short_avg - x[i]) > max_new:
                max_new = abs(short_avg_sum / samples_num_short_avg - x[i])
            search_samples_left -= 1
            if search_samples_left == 0:
                found_peaks.append(max_x)
                threshold = GAMMA * threshold + ALPHA * (1 - GAMMA) * max_new #TODO: z czego tu ten maks
                is_inside_searching_window = False
                is_inside_refractory_window = True
                refractory_window_end = max_x + samples_num_window
                max_x = 0
                max_new = 0.0
                max_abs_y = 0.0
    # print('Tape evaluated by alg4')
    return found_peaks


def alg4_polish_20210222(x):
    ALPHA = 0.45
    GAMMA = 0.98
    # samples_num_short_avg = 20
    # samples_num_long_avg = 100
    samples_num_short_avg = 17
    samples_num_long_avg = 30
    samples_num_window = 72
    max_diff_arg, max_diff_val = alg4_find_first_peak_3s(x)
    found_peaks = [max_diff_arg]
    threshold = ALPHA * max_diff_val
    short_avg_sum = 0.0
    long_avg_sum = 0.0
    search_samples_left = 0
    max_x = 0
    max_abs_y = 0.0
    refractory_window_end = found_peaks[0] + samples_num_window
    is_inside_refractory_window = True
    is_inside_searching_window = False
    x_len = len(x)
    max_new = 0.0
    for i in range(0, x_len):
        short_avg_sum += x[i]
        long_avg_sum += x[i]
        if i >= samples_num_short_avg:
            short_avg_sum -= x[i - samples_num_short_avg]
        if i >= samples_num_long_avg:
            long_avg_sum -= x[i - samples_num_long_avg]
        if i < samples_num_short_avg - 1:
            continue

        abs_diff_short = abs(short_avg_sum / samples_num_short_avg - x[i]) #TODO: czy tu na pewno ten short diff
        if is_inside_refractory_window:
            if i == refractory_window_end:
                is_inside_refractory_window = False
            else:
                continue
        if abs_diff_short >= threshold:
            if not is_inside_searching_window:
                is_inside_searching_window = True
                search_samples_left = samples_num_window
        if is_inside_searching_window:
            if abs(long_avg_sum / samples_num_long_avg - x[i]) > max_abs_y:
                max_x = i
                max_abs_y = abs(long_avg_sum / samples_num_long_avg - x[i])
                max_abs_short = abs_diff_short
            if abs(short_avg_sum / samples_num_short_avg - x[i]) > max_new:
                max_new = abs(short_avg_sum / samples_num_short_avg - x[i])
            search_samples_left -= 1
            if search_samples_left <= 0:
                # # if short_y[m] > peak_treshold or (abs(ref_tresholds[m] - ref_baseline[m])) > peak_treshold:
                # if search_samples_left == 0:
                #     threshold = GAMMA * threshold + ALPHA * (1 - GAMMA) * max_new  # TODO: z czego tu ten maks
                # # if abs_diff_short > threshold or (abs(short_avg_sum / samples_num_short_avg - x[i])) > threshold:
                # if abs(short_avg_sum / samples_num_short_avg - x[i]) > threshold:
                #     continue
                found_peaks.append(max_x)
                threshold = GAMMA * threshold + ALPHA * (1 - GAMMA) * max_new #TODO: z czego tu ten maks
                is_inside_searching_window = False
                is_inside_refractory_window = True
                refractory_window_end = max_x + samples_num_window
                max_x = 0
                max_new = 0.0
                max_abs_y = 0.0
    # print('Tape evaluated by alg4')
    return found_peaks



def alg4_find_first_peak(x):
    short_avg_sum = 0.0
    max_diff_val = -1.0
    max_diff_arg = 0
    for i in range(0, HZ):
        short_avg_sum += x[i]
        if i < 20 - 1:
            continue
        abs_diff_short = abs(short_avg_sum / 20 - x[i])
        if abs_diff_short > max_diff_val:
            max_diff_val = abs_diff_short
            max_diff_arg = i
        short_avg_sum -= x[i - 20]
    return max_diff_arg, max_diff_val


def alg4_find_first_peak_3s(x):
    short_avg_sum = 0.0
    max_diff_val = -1.0
    max_diff_arg = 0
    three_sec = int(3 * HZ)
    for i in range(0, three_sec):
        short_avg_sum += x[i]
        if i < 20 - 1:
            continue
        abs_diff_short = abs(short_avg_sum / 20 - x[i])
        if abs_diff_short > max_diff_val:
            max_diff_val = abs_diff_short
            max_diff_arg = i
        short_avg_sum -= x[i - 20]
    return max_diff_arg, max_diff_val



def alg5_pan_tompkins(x):
    return pan_tompkins_detector(x)

fs = 360
def pan_tompkins_detector(unfiltered_ecg, MWA_name='cumulative'):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * fs)
    mwa = MWA_from_name(MWA_name)(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "cumulative_opers":
        return MWA_cumulative_opers
    elif function_name == "cumulative_opers_no_counting":
        return MWA_cumulative_opers_no_counting
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else:
        raise RuntimeError('invalid moving average function!')


# Fast implementation of moving window average with numpy's cumsum function
def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


# Original Function
def MWA_original(input_array, window_size):
    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]

    for i in range(2, len(input_array) + 1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        mwa[i - 1] = np.mean(section)

    return mwa


# Fast moving window average implemented with 1D convolution
def MWA_convolve(input_array, window_size):
    ret = np.pad(input_array, (window_size - 1, 0), 'constant', constant_values=(0, 0))
    ret = np.convolve(ret, np.ones(window_size), 'valid')

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks


def my_diff(x, opers):
    if type(x) == np.ndarray:
        diff = x.tolist()
    else:
        diff = x
    last_idx = len(diff) - 1
    opers['SUB'] += 1
    for i in range(0, last_idx):
        opers['ADD'] += 1
        opers['SUB'] += 1
        diff[i] = diff[i + 1] - diff[i]
    return np.array(diff[:-1])


def my_argmax(x, opers):
    max_idx = 0
    for i in range(0, len(x)):
        opers['COMP'] += 1
        if x[max_idx] < x[i]:
            max_idx = i
    return max_idx


def my_mean(x, opers):
    suma = 0.0
    for i in x:
        opers['ADD'] += 1
        suma += i
    opers['DIV'] += 1
    return suma / len(x)


def MWA_cumulative_opers(input_array, window_size, opers):
    opers['ADD'] += len(input_array)
    ret = np.cumsum(input_array, dtype=float)
    opers['SUB'] += len(ret) - window_size
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        opers['DIV'] += 1
        opers['SUB'] += 2
        ret[i - 1] = ret[i - 1] / i
    opers['MUL'] += len(ret) - window_size
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def my_diff_no_counting(x):
    if type(x) == np.ndarray:
        diff = x.tolist()
    else:
        diff = x
    last_idx = len(diff) - 1
    for i in range(0, last_idx):
        diff[i] = diff[i + 1] - diff[i]
    return np.array(diff[:-1])


def MWA_cumulative_opers_no_counting(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def my_argmax_no_counting(x):
    max_idx = 0
    for i in range(0, len(x)):
        if x[max_idx] < x[i]:
            max_idx = i
    return max_idx


def my_mean_no_counting(x):
    suma = 0.0
    for i in x:
        suma += i
    return suma / len(x)



#conf.py
import os
from os import listdir
from os.path import isfile, join

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
#DB_DIR = ROOT_DIR + '/db'
#DBNS_DIR = ROOT_DIR + '/db_noise'

#ALL_FILES = list({f.split('.')[0] for f in listdir(DB_DIR) if isfile(join(DB_DIR, f))} - {'.'})

# R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e']
#R_SYMBOLS = ['N', 'V', 'L', 'R', '/', 'f', 'A', 'E', 'Q', 'F', 'j', 'J', 'a', 'S', 'e', 'r', 'F', 'n', '?']
HZ = 360

FILTER_207 = True
FILTERED_207 = [(14665, 18450),
                (19715, 21731),
                (87172, 88716),
                (89242, 94122),
                (97008, 101126),
                (554826, 589926)]

# pth = 6.6
# fs = 360
# SPANISH_TH_FACTOR = math.exp(-1 * (pth / fs))
SPANISH_TH_FACTOR = 0.98183



# Hamilton
from collections import deque
from bisect import insort

fs = 360

def hamilton_detector(unfiltered_ecg):
    """
        P.S. Hamilton, 
        Open Source ECG Analysis Software Documentation, E.P.Limited, 2002.
    """
        
    f1 = 8/fs
    f2 = 16/fs

    b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

    diff = abs(np.diff(filtered_ecg))

    b = np.ones(int(0.08*fs))
    b = b/int(0.08*fs)
    a = [1]

    ma = signal.lfilter(b, a, diff)

    ma[0:len(b)*2] = 0

    n_pks = deque([], maxlen=8)
    n_pks_ave = 0.0
    s_pks = deque([], maxlen=8)
    s_pks_ave = 0.0
    QRS = [0]
    RR = deque([], maxlen=8)
    RR_ave = 0.0

    th = 0.0

    i=0
    idx = []
    peaks = []  

    for i in range(1, len(ma) - 1):
        if ma[i - 1] < ma[i] and ma[i + 1] < ma[i]:
            peak = i
            peaks.append(i)
            if ma[peak] > th and (peak-QRS[-1])>0.3*fs:        
                QRS.append(peak)
                idx.append(i)
                s_pks.append(ma[peak])
                s_pks_ave = np.mean(s_pks)

                if (RR_ave != 0.0) and (QRS[-1]-QRS[-2] > 1.5*RR_ave):
                    missed_peaks = peaks[idx[-2]+1:idx[-1]]
                    for missed_peak in missed_peaks:
                        if missed_peak-peaks[idx[-2]]>int(0.360*fs) and ma[missed_peak]>0.5*th:
                            insort(QRS, missed_peak)
                            break

                if len(QRS)>2:
                    RR.append(QRS[-1]-QRS[-2])
                    RR_ave = int(np.mean(RR))

            else:
                n_pks.append(ma[peak])
                n_pks_ave = np.mean(n_pks)

            th = n_pks_ave + 0.45*(s_pks_ave-n_pks_ave)

    QRS.pop(0)

    return QRS



# Add fifth algorithm, JP
N = int(360*0.15)
K = 6
R_si = int(360*0.24)
R_si_2 = int(360*0.12)

def Alg5_Turkish(x):
    
    N = int(360*0.15)
    K = 6
    R_si = int(360*0.24)
    R_si_2 = int(360*0.12)

    x = np.array(x)
    # Digital FIR, y0 = x(n) - x(n-K), where K=f_sample/f_line
    K = 6
    y0 = x[K:] - x[:-K]
    
    # Squaring
    y1 = y0**2

    # Moving average
    window = np.ones(N) / N
    y2 = np.convolve(y1, window, mode='valid')

    # Normalization
    y2_min = np.min(y2)
    y2_max = np.max(y2)
    y = (y2 - y2_min) / (y2_max - y2_min)

    # adaptive tresholding
    thv = np.zeros_like(y, dtype=float)  # Initialize thv array
    
    # Calculate the treshold values
    thv[0] = 1
    for n in range(1, len(y)):
        thv[n] = ((N - 1) * thv[n - 1] + y[n]) / N

    # QRS occurs whenever signal is greater than the treshold
    temp_loc = np.where(y > thv)[0]

    R_peaks = [temp_loc[0]]

    for i in range(1, len(temp_loc)):
        if temp_loc[i] - temp_loc[i-1] > R_si:
            # I've added to every detection, so not the first value that passed treshold is considered
            R_peaks.append(temp_loc[i]+R_si + 20)
            # R_peaks.append(temp_loc[i])


    # Instead of adding 'R_si + 20' authors proposed a method for finding max (or min) value in a given window
    # to detect R-peak, but it's giving similar results and is usless from the classification point of viev

    # L = 20
    # R_peak_max = []

    # for peak in R_peaks:
    #     if np.mean(x[peak-L:peak+L]) > 0:
    #         R_peak_max.append(np.argmax(x[peak - R_si_2 : peak + R_si_2]) + peak - R_si_2)
    #     else:
    #         R_peak_max.append(np.argmin(x[peak - R_si//2 : peak + R_si//2]) + peak - R_si//2)


    # return R_peak_max

    return R_peaks

