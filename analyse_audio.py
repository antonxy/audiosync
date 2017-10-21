# Contains code form https://github.com/kit-cel/lecture-examples/blob/master/nt1/vorlesung marked with EXTERNAL licensed under GPL

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

if False:
    def decode_signal(data, chunk_length, frequency, signal_start, sample_rate, num_bits):
        """
        decodes the n byte BPSK-signal in the audio data
        """
        max_arg = int(frequency * chunk_length * 2 * np.pi)
        one_chunk = np.sin(np.linspace(0, max_arg, chunk_length * sample_rate))

        chunk_samples = int(sample_rate * chunk_length)
        decoded_data = []

        for delta_n in range(signal_start, signal_start + chunk_samples * (num_bits + 1), chunk_samples):
            if delta_n + chunk_samples > data.size:
                break

            current_chunk = data[delta_n:delta_n + chunk_samples]
            prod = np.multiply(current_chunk, one_chunk)
            s = np.sum(prod)
            decoded_data.append(1 if s > 0 else 0)

        return decoded_data[1:]
else:

    def decode_signal(data, chunk_length, frequency, signal_start, sample_rate, num_bits):
        """
        decodes the n byte BPSK-signal in the audio data
        """
        chunk_samples = int(sample_rate * chunk_length)
        decoded_data = []
        last_chunk = None
        last_bit = 0
        #                                                                Read n bytes + start bit
        for delta_n in range(signal_start, signal_start + chunk_samples * (num_bits + 1), chunk_samples):
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
        return decoded_data


def find_and_decode_signal(data, sample_rate, chunk_length, frequency, chirp_f0, chirp_f1, chirp_duration):
    """
    returns: signal start, decoded bytes
    """
    data = bandpass(data, frequency, 500, sample_rate)
    signal_start = find_sync_signal(data, sample_rate, generate_chirp(chirp_f0, chirp_f1, chirp_duration, sample_rate))
    decoded_data = bits_to_bytes(decode_signal(data, chunk_length, frequency, signal_start, sample_rate, 32))
    return signal_start, decoded_data


def generate_signal(bits, chunk_length, sample_rate, frequency):
    #max_arg = int(frequency * chunk_length / sample_rate * 2 * np.pi)
    max_arg = int(frequency * chunk_length * 2 * np.pi)

    zero_chunk = np.sin(np.linspace(0 + np.pi, max_arg + np.pi, chunk_length * sample_rate))
    one_chunk = np.sin(np.linspace(0, max_arg, chunk_length * sample_rate))

    data = np.zeros(0)

    for bit in bits:
        if bit == 1:
            data = np.append(data, one_chunk)
        else:
            data = np.append(data, zero_chunk)

    return data

# EXTERNAL
########################
# find impulse response of an RC filter
########################
def get_rc_ir(K, n_up, t_symbol, r):
    
    ''' 
    Determines coefficients of an RC filter 
    
    Formula out of: K.-D. Kammeyer, Nachrichtenübertragung
    At poles, l'Hospital was used 
    
    NOTE: Length of the IR has to be an odd number
    
    IN: length of IR, upsampling factor, symbol time, roll-off factor
    OUT: filter coefficients
    '''

    # check that IR length is odd
    assert K % 2 == 1, 'Length of the impulse response should be an odd number'
    
    # map zero beta to close-to-zero
    if r == 0:
        r = 1e-32


    # initialize output length and sample time
    rc = np.zeros( K )
    t_sample = t_symbol / n_up
    
    
    # time indices and sampled time
    k_steps = np.arange( -(K-1) / 2.0, (K-1) / 2.0 + 1 )   
    t_steps = k_steps * t_sample
    
    for k in k_steps.astype(int):
        
        if t_steps[k] == 0:
            rc[ k ] = 1.
            
        elif np.abs( t_steps[k] ) == t_symbol / ( 2.0 * r ):
            rc[k] = np.sin(np.pi/(2*r)) / (np.pi/(2*r)) * np.pi / 4
        #    rc[ k ] = r * np.sin( np.pi / r )
        #    
        else:
            rc[ k ] = np.sin( np.pi * t_steps[k]/t_symbol ) / (np.pi * t_steps[k]/t_symbol) \
                * np.cos( r * np.pi * t_steps[k] / t_symbol ) \
                / ( 1.0 - ( 2.0 * r * t_steps[k] / t_symbol )**2 )
        #    rc[ k ] = np.sin( np.pi * t_steps[k]/t_symbol ) / np.pi / t_steps[k] \
        #        * np.cos( r * np.pi * t_steps[k] / t_symbol ) \
        #        / ( 1.0 - ( 2.0 * r * t_steps[k] / t_symbol )**2 )
 
    return rc


########################
# find impulse response of an RRC filter
########################
def get_rrc_ir(K, n_up, t_symb, r):
        
    ''' 
    Determines coefficients of an RRC filter 
    
    Formula out of: J. Huber, Trelliscodierung, Springer, 1992, S. 15
    At poles, values of wikipedia.de were used (without cross-checking)
    
    NOTE: Length of the IR has to be an odd number
    
    IN: length of IR, upsampling factor, symbol time, roll-off factor
    OUT: filter ceofficients
    '''

    assert K % 2 != 0, "Filter length needs to be odd"
    
    if r == 0:
        r = 1e-32

    # init
    rrc = np.zeros(K)
    t_sample = t_symb/n_up
    
        
    i_steps = np.arange( 0, K)
    k_steps = np.arange( -(K-1)/2.0, (K-1)/2.0 + 1 )    
    t_steps = k_steps*t_sample

    for i in i_steps:

        if t_steps[i] == 0:
            rrc[i] = 1.0/np.sqrt(t_symb) * (1.0 - r + 4.0 * r / np.pi )

        elif np.abs( t_steps[i] ) == t_symb/4.0/r:
            rrc[i] = r/np.sqrt(2.0*t_symb)*((1+2/np.pi)*np.sin(np.pi/4.0/r)+ \
                            ( 1.0 - 2.0/np.pi ) * np.cos(np.pi/4.0/r) )

        else:
            rrc[i] = 1.0/np.sqrt(t_symb)*( np.sin( np.pi*t_steps[i]/t_symb*(1-r) ) + \
                            4.0*r*t_steps[i]/t_symb * np.cos( np.pi*t_steps[i]/t_symb*(1+r) ) ) \
                            / (np.pi*t_steps[i]/t_symb*(1.0-(4.0*r*t_steps[i]/t_symb)**2.0))
 
    return rrc
# END EXTERNAL

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# EXTERNAL
def generate_constellation_points(M):
    return [ np.exp( 1j * 2 * np.pi * m / M + 1j * np.pi / M ) for m in range( M ) ]
# END EXTERNAL

def generate_rc_filter(n_up, symbol_time):
    syms_per_filt = 4
    K_filt = 2 * syms_per_filt * n_up + 1
    rc = get_rc_ir(K_filt, n_up, symbol_time, 0.33)
    return rc / np.linalg.norm(rc)

def generate_rrc_filter(n_up, symbol_time):
    syms_per_filt = 4
    K_filt = 2 * syms_per_filt * n_up + 1
    rrc = get_rrc_ir(K_filt, n_up, symbol_time, 0.33)
    #return rrc / np.linalg.norm(rrc)
    return rrc / rrc[int(rrc.shape[0] / 2)]

def generate_signal_new(symbols, n_up, sr, frequency):
    M = 4
    constellation_points = generate_constellation_points(M)

    plt.plot(np.real(constellation_points), np.imag(constellation_points), 'o')
    for i, xy in enumerate(zip(np.real(constellation_points), np.imag(constellation_points))):
        plt.annotate(str(i), xy=xy, textcoords='data')
    plt.show()

    s = [ constellation_points[symbol] for symbol in symbols ]
    n_symbols = len(s)
    symbol_time = n_up / sr

    rect = np.ones(n_up)

    rrc = generate_rrc_filter(n_up, symbol_time) * 0.7

    plt.plot(rrc)
    print(rrc.shape)

# EXTERNAL
    s_up = np.zeros(n_symbols * n_up, dtype=np.complex128)
    s_up[::n_up] = s
    s_up = np.convolve(rrc, s_up)

    plt.plot(np.abs(s_up))
    plt.show()

    plt.plot( np.real( s_up ), np.imag( s_up ), linewidth=2.0 )
    plt.grid( True )
    plt.xlabel( '$\mathrm{Re}\\{s(t)\\}$' )
    plt.ylabel(' $\mathrm{Im}\\{s(t)\\}$' )
    plt.title( 'QPSK signal' )
    plt.show()
# END EXTERNAL

    t = np.linspace(0, s_up.shape[0]/sr, s_up.shape[0])
    carrier = np.exp(-1j*2*np.pi*frequency*t)
    
    modulated = np.real(s_up * carrier)
    print(modulated.shape)

    #plt.plot(modulated)
    #plt.show()

    return modulated

def demodulate_signal(signal, n_data_symbols, n_up, sr, frequency):
    M = 4
    constellation_points = generate_constellation_points(M)
    symbol_time = n_up / sr
    n_symbols = n_data_symbols + 8 + 1  # 8 for rc, 1 for differential

    signal = signal[:n_symbols * n_up]

    t = np.linspace(0, n_symbols * symbol_time, n_symbols * n_up)
    dem_carrier = np.exp(1j*2*np.pi*frequency*t)

    demodulated = signal * dem_carrier

    rrc = generate_rrc_filter(n_up, symbol_time)
    rrc /= np.linalg.norm(rrc)
    demodulated = np.convolve(demodulated, rrc[::-1])

    sampled = demodulated[::n_up]
    sampled = sampled[8:-8] #remove 2x rc length at sides

    diff_dec = differential_decode_symbols(sampled)


# EXTERNAL
    plt.plot( np.real( demodulated ), np.imag( demodulated ), linewidth=2.0 )
    plt.grid( True )
    plt.xlabel( '$\mathrm{Re}\\{s(t)\\}$' )
    plt.ylabel(' $\mathrm{Im}\\{s(t)\\}$' )
    plt.title( 'Demodulated QPSK signal' )
    plt.show()

    plt.plot( np.real( diff_dec ), np.imag( diff_dec ), 'o', linewidth=2.0 )
    for i, xy in enumerate(zip(np.real(diff_dec), np.imag(diff_dec))):
        plt.annotate(str(i), xy=xy, textcoords='data')
    plt.grid( True )
    plt.xlabel( '$\mathrm{Re}\\{s(t)\\}$' )
    plt.ylabel(' $\mathrm{Im}\\{s(t)\\}$' )
    plt.title( 'Demodulated QPSK signal' )
    plt.show()
# END EXTERNAL

    symbols = []
    for s in diff_dec:
        upper = np.imag(s) > 0
        right = np.real(s) > 0
        if upper and right:
            symbols.append(0)
        if upper and not right:
            symbols.append(1)
        if not upper and not right:
            symbols.append(2)
        if not upper and right:
            symbols.append(3)

    return symbols


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

def differential_encode_symbol_nums(symbol_nums, M):
    encoded = [0]
    last = 0
    for s in symbol_nums:
        enc = (s + last) % M
        last = enc
        encoded.append(enc)
    return encoded


def differential_decode_symbols(symbols):
    last = symbols[0]
    decoded = np.zeros(symbols.shape[0] - 1, dtype=np.complex128)
    for i, s in enumerate(symbols[1:]):
        decoded[i] = s / last * (1+1j)
        last = s
    return decoded


def find_sync_signal(data, sr, sync_signal):
    """
    returns: sample at the end of the sync signal
    """
    res = scipy_signal.fftconvolve(data, sync_signal[::-1], 'valid')
    return np.argmax(np.abs(res)) + sync_signal.size


def bandpass(data, f, pass_width, sr):
    filter_array = scipy_signal.firwin(128, [f-pass_width, f+pass_width], pass_zero=False, nyq=sr/2)
    data2 = scipy_signal.fftconvolve(data, filter_array, mode='same')
    return data2

if __name__ == "__main__":
    import scipy_wavfile as wavfile
    import random

    random.seed(1337)
    data = [random.randint(0, 3) for _ in range(64)]

    print([9] + data)

    diff_data = differential_encode_symbol_nums(data, 4)
    print(diff_data)

    sync_signal = generate_chirp(2000, 4000, 0.3, 48000)

    data_signal = generate_signal_new(diff_data, 256, 48000, 4000)
    signal = np.append(sync_signal, data_signal)
    wavfile.write("test_new.wav", 48000, (signal * np.iinfo(np.int16).max).astype(np.int16))

    sr, signal = wavfile.read("test_new_rec_rcc_bad.wav")
    assert sr == 48000
    signal = (signal[:,0]).astype(np.float) / np.iinfo(np.int16).max
    print("read length ", signal.shape)

    end_of_sync = find_sync_signal(signal, 48000, sync_signal)
    print("end of sync: ", end_of_sync)
    demod_data = demodulate_signal(signal[end_of_sync:], 64, 256, 48000, 4000)
    ser = sum([int(orig != demod) for orig, demod in zip(data, demod_data)])/len(data)

    print(data)
    print(len(data))
    print(demod_data)
    print(len(demod_data))
    print("SER: {}".format(ser))
