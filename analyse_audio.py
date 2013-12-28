import numpy as np
import scipy.signal


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


def decode_signal(data, chunk_length, signal_start, sample_rate, num_bytes):
    """
    decodes the n byte BPSK-signal in the audio data
    """
    chunk_samples = int(sample_rate * chunk_length)
    decoded_data = []
    last_chunk = None
    last_bit = 0
    #                                                                Read n bytes + start bit
    for delta_n in range(signal_start, signal_start + chunk_samples * (num_bytes * 8 + 1), chunk_samples):
        if delta_n + chunk_samples > data.size:
            break

        current_chunk = data[delta_n:delta_n + chunk_samples]
        if last_chunk is not None:
            prod = np.multiply(current_chunk, last_chunk)
            s = np.sum(prod)

            #Switch bit if s < 0; s < 0 means phase change
            if s < 0:
                if last_bit < 1:
                    last_bit = 1
                else:
                    last_bit = 0
            decoded_data.append(last_bit)

        last_chunk = current_chunk
    return bits_to_bytes(decoded_data)


def find_and_decode_signal(data, sample_rate, chunk_length, chirp_f0, chirp_f1, chirp_duration):
    """
    returns: signal start, decoded bytes
    """
    signal_start = find_sync_signal(data, sample_rate, generate_chirp(chirp_f0, chirp_f1, chirp_duration, sample_rate))
    decoded_data = decode_signal(data, chunk_length, signal_start, sample_rate, 4)
    return signal_start, decoded_data


def generate_signal(bits, chunk_length, sample_rate, frequency):
    max_arg = int(frequency * chunk_length / sample_rate * 2 * np.pi)

    zero_chunk = np.sin(np.linspace(0 + np.pi, max_arg + np.pi, chunk_length * sample_rate))
    one_chunk = np.sin(np.linspace(0, max_arg, chunk_length * sample_rate))

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
    return scipy.signal.chirp(ts, f0, duration, f1, method='linear')


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
    res = scipy.signal.fftconvolve(data, sync_signal[::-1], 'valid')
    return np.argmax(np.abs(res)) + sync_signal.size