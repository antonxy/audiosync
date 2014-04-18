import numpy as np
import scipy_signal


def bits_to_bytes(bits):
    bs = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        bs.append(int(''.join([str(bit) for bit in byte]), 2))
    return bs


def find_signal(data, freq, sr):
    """
    finds the beginning of a certain frequency in the audio data
    """
    #                 oscillations / sec * seconds      * rad / oscillation
    max_cos_arg = int(freq * data.size / sr * 2 * np.pi)
    cmp_cos = np.cos(np.linspace(0, max_cos_arg, data.size))
    mul = np.abs(np.multiply(data, cmp_cos))
    th = np.amax(mul)/2
    return np.argmax(np.greater(mul, th))


def decode_signal(data, chunk_length, f0, f1, signal_start, sample_rate, num_bits):
    """
    decodes the n byte BPSK-signal in the audio data
    """
    chunk_samples = int(sample_rate * chunk_length)
    decoded_data = np.array([])

    max_arg_0 = int(f0 * chunk_length * 2 * np.pi)
    max_arg_1 = int(f1 * chunk_length * 2 * np.pi)

    zero_chunk = np.exp(1j * np.linspace(0, max_arg_0, chunk_samples))
    one_chunk = np.exp(1j * np.linspace(0, max_arg_1, chunk_samples))

    #                                                                Read n bytes + start bit
    for delta_n in range(signal_start, signal_start + chunk_samples * (num_bits + 1), chunk_samples):
        if delta_n + chunk_samples > data.size:
            break

        current_chunk = data[delta_n:delta_n + chunk_samples]
        prod0 = np.multiply(current_chunk, zero_chunk)
        s0 = np.sum(prod0)

        prod1 = np.multiply(current_chunk, one_chunk)
        s1 = np.sum(prod1)

        decoded_data = np.append(decoded_data, np.abs(s1) > np.abs(s0))
    return decoded_data.astype(int)


def find_and_decode_signal(data, sample_rate, chunk_length, frequency0, frequency1, chirp_f0, chirp_f1, chirp_duration):
    """
    returns: signal start, decoded bytes
    """
    signal_start = find_sync_signal(data, sample_rate, generate_chirp(chirp_f0, chirp_f1, chirp_duration, sample_rate))
    decoded_data = bits_to_bytes(decode_signal(data, chunk_length, frequency0, frequency1, signal_start, sample_rate, 32))
    return signal_start, decoded_data


def generate_signal(bits, chunk_length, sample_rate, f0, f1):
    #max_arg = int(frequency * chunk_length / sample_rate * 2 * np.pi)
    max_arg_0 = int(f0 * chunk_length * 2 * np.pi)
    max_arg_1 = int(f1 * chunk_length * 2 * np.pi)

    zero_chunk = np.sin(np.linspace(0, max_arg_0, chunk_length * sample_rate))
    one_chunk = np.sin(np.linspace(0, max_arg_1, chunk_length * sample_rate))

    data = np.zeros(0)

    for bit in bits:
        if bit == 1:
            data = np.append(data, one_chunk)
        else:
            data = np.append(data, zero_chunk)

    return data


def generate_chirp(f0, f1, duration, sample_rate):
    ts = np.linspace(0, duration, duration * sample_rate)
    #linear cosine chirp
    return scipy_signal.chirp(ts, f0, duration, f1, method='linear')


def check_checksum(data):
    checksum = 0
    for i in data[0:3]:
        checksum += i

    checksum %= 255

    return checksum == data[3]


def find_sync_signal(data, sr, sync_signal):
    """
    returns: sample at the end of the sync signal
    """
    res = scipy_signal.fftconvolve(data, sync_signal[::-1], 'valid')
    return np.argmax(np.abs(res)) + sync_signal.size