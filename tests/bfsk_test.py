import analyse_audio
import numpy as np
import matplotlib.pyplot as plt


def bfsk_single_test(chunk_length, f0, f1, noise_factor=0, shift_error=0):
        bits = np.random.choice([0, 1], 32)
        signal = analyse_audio.generate_signal(bits, chunk_length, 48000, f0, f1)
        shift_signal = np.zeros(shift_error)
        signal = np.append(shift_signal, signal)
        if noise_factor != 0:
            noise = np.random.random(signal.size) * noise_factor
            noised_signal = np.add(signal, noise)
        else:
            noised_signal = signal

        decoded_bits = analyse_audio.decode_signal(noised_signal, chunk_length, f0, f1, 0, 48000, len(bits))
        if len(bits) != len(decoded_bits):
            return False
        return (bits == np.array(decoded_bits)).all()


def bfsk_noise_test():
    n = 200
    chunk_length = 0.01
    f0, f1 = 3000, 6000
    noise_level_list = []
    success_list = []
    for noise_factor in np.linspace(0, 30, 20):
        success_ratio = 0
        for i in range(n):
            if bfsk_single_test(chunk_length, f0, f1, noise_factor=noise_factor):
                success_ratio += 1 / n
        noise_level_list.append(noise_factor)
        success_list.append(success_ratio)

        print('{} Successes at noise level {}'.format(success_ratio, noise_factor))

    plt.plot(noise_level_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Chunk length: {}, f0: {}, f1: {}'.format(n, chunk_length, f0, f1))
    plt.ylabel('Success ratio')
    plt.xlabel('Noise level')
    plt.show()


def bfsk_chunk_length_test():
    n = 50
    noise_level = 10
    f0, f1 = 3000, 6000
    chunk_length_list = []
    success_list = []
    for chunk_length in np.logspace(-4, -1.9, 20):
        success_ratio = 0
        for i in range(n):
            if bfsk_single_test(chunk_length, f0, f1, noise_factor=noise_level):
                success_ratio += 1 / n
        chunk_length_list.append(chunk_length)
        success_list.append(success_ratio)

        print('{} Successes at length {}'.format(success_ratio, chunk_length))

    plt.plot(chunk_length_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, f0: {}, f1: {}'.format(n, noise_level, f0, f1))
    plt.ylabel('Success ratio')
    plt.xlabel('Chunk length [s]')
    plt.show()


def bfsk_shift_error_test():
    n = 50
    noise_level = 2
    chunk_length = 0.01
    f0, f1 = 3000, 6000
    shift_error_list = []
    success_list = []
    for shift_error in np.logspace(1, 3, 20):
        success_ratio = 0
        for i in range(n):
            if bfsk_single_test(chunk_length, f0, f1, noise_factor=noise_level, shift_error=shift_error):
                success_ratio += 1 / n
        shift_error_list.append(shift_error)
        success_list.append(success_ratio)

        print('{} Successes at Shift Error {}'.format(success_ratio, shift_error))

    plt.plot(shift_error_list, success_list, 'bo-')
    plt.title('Trials per value: {}, Noise level: {}, Chunk length: {}, f0: {}, f1: {}'.
              format(n, noise_level, chunk_length, f0, f1))
    plt.ylabel('Success ratio')
    plt.xlabel('Shift error [samples]')
    plt.show()

if __name__ == '__main__':
    bfsk_shift_error_test()