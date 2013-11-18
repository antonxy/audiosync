import numpy as np


def bits_to_bytes(bits):
    bs = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        bs.append(int(''.join([str(bit) for bit in byte]), 2))
    return bs


def find_signal(data, freq, sr):
    '''
    finds the beginning of a certain frequency in the audio data
    '''
    #                 oscillations / sec * seconds      * rad / oscillation
    max_cos_arg = int(freq * data.size / sr * 2 * np.pi)
    cmp_cos = np.cos(np.linspace(0, max_cos_arg, data.size))
    mul = np.abs(np.multiply(data, cmp_cos))
    th = np.amax(mul)/2
    return np.argmax(np.greater(mul, th))


def decode_signal(data, chunk_length, signal_start, sample_rate, num_bytes):
    '''
    decodes the n byte BPSK-signal in the audio data
    '''
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


def find_and_decode_signal(data, sample_rate, carrier_frequency, chunk_length):
    signal_start = find_signal(data, carrier_frequency, sample_rate)
    decoded_data = decode_signal(data, chunk_length, signal_start, sample_rate, 3)
    return decoded_data