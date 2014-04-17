import analyse_audio
import numpy as np
import matplotlib.pyplot as plt


def chirp_single_test(length, freq0, freq1, noise_factor=0):
        chirp = analyse_audio.generate_chirp(freq0, freq1, length, 48000)
        zeros = np.zeros(chirp.size)
        signal = np.append(np.append(zeros, chirp), zeros)
        if noise_factor != 0:
            noise = np.random.random(signal.size) * noise_factor
            noised_signal = np.add(signal, noise)
        else:
            noised_signal = signal

        expected = chirp.size * 2
        result = analyse_audio.find_sync_signal(noised_signal, 48000, analyse_audio.generate_chirp(freq0, freq1, length, 48000))
        return result - expected


def chirp_length_test():
    n = 20
    noise_level = 1.5
    f0 = 3000
    f1 = 6000
    length_list = []
    std_dev_list = []
    max_dev_list = []
    for length in np.logspace(-4, -1, 20):
        samples = np.array([])
        for i in range(n):
            samples = np.append(samples, chirp_single_test(length, f0, f1, noise_level))
        length_list.append(length)
        std_dev = np.sqrt(np.sum(np.square(samples)) / samples.size)
        std_dev_list.append(std_dev)
        max_dev_list.append(np.amax(np.abs(samples)))

        print('Std dev {} at length {}'.format(std_dev, length))

    plt.plot(length_list, std_dev_list, 'bo-')
    plt.plot(length_list, max_dev_list, 'ro-')
    plt.title('Trials per value: {}, Noise level: {}, f0: {}, f1: {}'.format(n, noise_level, f0, f1))
    plt.ylabel('std dev [samples]')
    plt.xlabel('length [s]')
    plt.show()


def chirp_noise_test():
    n = 50
    length = 0.1
    f0 = 3000
    f1 = 6000
    noise_level_list = []
    std_dev_list = []
    max_dev_list = []
    for noise_level in np.linspace(0, 50, 20):
        samples = np.array([])
        for i in range(n):
            samples = np.append(samples, chirp_single_test(length, f0, f1, noise_level))
        noise_level_list.append(noise_level)
        std_dev = np.sqrt(np.sum(np.square(samples)) / samples.size)
        std_dev_list.append(std_dev)
        max_dev_list.append(np.amax(np.abs(samples)))

        print('Std dev {} at noise level {}'.format(std_dev, noise_level))

    plt.plot(noise_level_list, std_dev_list, 'bo-')
    plt.plot(noise_level_list, max_dev_list, 'ro-')
    plt.title('Trials per value: {}, Noise level: {}, f0: {}, f1: {}'.format(n, noise_level, f0, f1))
    plt.ylabel('std dev [samples]')
    plt.xlabel('noise level')
    plt.show()

if __name__ == '__main__':
    chirp_noise_test()