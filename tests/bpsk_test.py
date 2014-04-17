import analyse_audio
import numpy as np
import matplotlib.pyplot as plt


def bpsk_single_test(chunk_length, frequency, noise_factor=0, frequency_error=0, shift_error=0):
        bits = np.random.choice([0, 1], 32)
        signal = analyse_audio.generate_signal(np.append([0], bits), chunk_length, 48000, frequency + frequency_error)
        shift_signal = np.zeros(shift_error)
        signal = np.append(shift_signal, signal)
        if noise_factor != 0:
            noise = np.random.random(signal.size) * noise_factor
            noised_signal = np.add(signal, noise)
        else:
            noised_signal = signal

        decoded_bits = analyse_audio.decode_signal(noised_signal, chunk_length, frequency, 0, 48000, len(bits))
        return (bits == np.array(decoded_bits)).all()


def bpsk_noise_test():
    n = 200
    chunk_length = 0.01
    frequency = 1000
    noise_level_list = []
    success_list = []
    for noise_factor in np.linspace(0, 5, 20):
        success_ratio = 0
        for i in range(n):
            if bpsk_single_test(chunk_length, frequency, noise_factor):
                success_ratio += 1 / n
        noise_level_list.append(noise_factor)
        success_list.append(success_ratio)

        print('{} Successes at noise level {}'.format(success_ratio, noise_factor))

    plt.plot(noise_level_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Chunk length: {}, Frequency: {}'.format(n, chunk_length, frequency))
    plt.ylabel('Success ratio')
    plt.xlabel('Noise level')
    plt.show()


def bpsk_chunk_length_test():
    n = 50
    noise_level = 10
    frequency = 6000
    chunk_length_list = []
    success_list = []
    for chunk_length in np.logspace(-4, -1.9, 20):
        success_ratio = 0
        for i in range(n):
            if bpsk_single_test(chunk_length, frequency, noise_level):
                success_ratio += 1 / n
        chunk_length_list.append(chunk_length)
        success_list.append(success_ratio)

        print('{} Successes at length {}'.format(success_ratio, chunk_length))

    plt.plot(chunk_length_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, Frequency: {}'.format(n, noise_level, frequency))
    plt.ylabel('Success ratio')
    plt.xlabel('Chunk length [s]')
    plt.show()


def bpsk_frequency_test():
    n = 50
    noise_level = 20
    chunk_length = 0.01
    freq_list = []
    success_list = []
    for freq in np.logspace(1, 4.5, 50):
        success_ratio = 0
        for i in range(n):
            if bpsk_single_test(chunk_length, freq, noise_level):
                success_ratio += 1 / n
        freq_list.append(freq)
        success_list.append(success_ratio)

        print('{} Successes at Frequency {}'.format(success_ratio, freq))

    plt.plot(freq_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, Chunk length: {}'.format(n, noise_level, chunk_length))
    plt.ylabel('Success ratio')
    plt.xlabel('Frequency [Hz]')
    plt.show()


def bpsk_frequency_error_test():
    n = 50
    noise_level = 2
    chunk_length = 0.01
    freq = 6000
    freq_error_list = []
    success_list = []
    for freq_error in np.logspace(0, 2.5, 20):
        success_ratio = 0
        for i in range(n):
            if bpsk_single_test(chunk_length, freq, noise_level, freq_error):
                success_ratio += 1 / n
        freq_error_list.append(freq_error)
        success_list.append(success_ratio)

        print('{} Successes at Frequency Error {}'.format(success_ratio, freq_error))

    plt.plot(freq_error_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, Chunk length: {}, Freq: {}'.
              format(n, noise_level, chunk_length, freq))
    plt.ylabel('Success ratio')
    plt.xlabel('Frequency error [Hz]')
    plt.show()


def bpsk_shift_error_test():
    n = 50
    noise_level = 2
    chunk_length = 0.01
    freq = 1000
    shift_error_list = []
    success_list = []
    for shift_error in np.logspace(0, 2, 20):
        success_ratio = 0
        for i in range(n):
            if bpsk_single_test(chunk_length, freq, noise_level, shift_error=shift_error):
                success_ratio += 1 / n
        shift_error_list.append(shift_error)
        success_list.append(success_ratio)

        print('{} Successes at Shift Error {}'.format(success_ratio, shift_error))

    plt.plot(shift_error_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, Chunk length: {}, Freq: {}'.
              format(n, noise_level, chunk_length, freq))
    plt.ylabel('Success ratio')
    plt.xlabel('Shift error [samples]')
    plt.show()

if __name__ == '__main__':
    bpsk_noise_test()