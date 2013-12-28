import read_audio
import analyse_audio
import os


def analyse_file(path):
    try:
        sr, audio = read_audio.from_file_normalized(path)
    except Exception:
        print('Could not open file %s' % path)
        return

    length = audio.size
    sync_point, data = analyse_audio.find_and_decode_signal(audio, sr, 0.05, 3000, 6000, 0.5)
    valid = analyse_audio.check_checksum(data)

    print('path {} analysed'.format(path))

    if valid:
        return {'path': path, 'sync_point_samples': sync_point,
                        'sst': data[0:3], 'sample_rate': sr, 'length_samples': length}


def rename_files(file_list, suffix):
    for e in file_list:
        file_ext = os.path.splitext(os.path.basename(e['path']))[-1]
        sst = e['sst']
        os.rename(e['path'], os.path.join(os.path.dirname(e['path']),
                                          '%i-%i-%i_%s%s' % (sst[0], sst[1], sst[2], suffix, file_ext)))


def generate_edls(videos, audios, fps, edl_dir):
    for v in videos:
        for a in audios:
            if v['sst'] == a['sst']:
                generate_edl(v, a, fps, edl_dir)


def generate_tc(seconds, fps):
    rem_seconds = float(seconds)
    hrs = int(rem_seconds // 3600)
    rem_seconds -= hrs * 3600
    mins = int(rem_seconds // 60)
    rem_seconds -= mins * 60
    secs = int(rem_seconds // 1)
    rem_seconds -= secs
    frames = round(rem_seconds * fps)

    return '{:02d}:{:02d}:{:02d}:{:02d}'.format(hrs, mins, secs, frames)


def generate_edl(video_data, audio_data, fps, edl_dir):
    sst = video_data['sst']
    sst_str = '-'.join(['%s' % e for e in sst])
    filename = sst_str + '.edl'

    f = open(os.path.join(os.path.abspath(edl_dir), filename), 'w')
    print('generating {}'.format(filename))

    #                  sample        /       sample_rate
    sync_sec_a = float(audio_data['sync_point_samples'] / float(audio_data['sample_rate']))
    sync_sec_v = float(video_data['sync_point_samples'] / float(video_data['sample_rate']))

    len_sec_a = float(audio_data['length_samples'] / float(audio_data['sample_rate']))
    len_sec_v = float(video_data['length_samples'] / float(video_data['sample_rate']))

    a_bef_v = sync_sec_a - sync_sec_v
    v_bef_a = -a_bef_v

    f.write('TITLE: %s   FORMAT: CMX3600\n' % sst_str)
    f.write('FCM: NON-DROP FRAME\n')
    if a_bef_v > 0:
        tc_v_start = generate_tc(a_bef_v, fps)
        tc_v_stop = generate_tc(a_bef_v + len_sec_v, fps)
        tc_a_len = generate_tc(len_sec_a, fps)
        tc_v_len = generate_tc(len_sec_v, fps)
        f.write('001  BL         V    C         00:00:00:00 {} 00:00:00:00 {}\n'.format(tc_v_start, tc_v_start))
        f.write('002  {:10s} V    C         00:00:00:00 {} {} {}\n'.format(sst_str + '_v', tc_v_len, tc_v_start, tc_v_stop))
        f.write('003  {:10s} AA   C         00:00:00:00 {} 00:00:00:00 {}\n'.format(sst_str + '_a', tc_a_len, tc_a_len))

    else:
        tc_a_start = generate_tc(v_bef_a, fps)
        tc_a_stop = generate_tc(v_bef_a + len_sec_a, fps)
        tc_a_len = generate_tc(len_sec_a, fps)
        tc_v_len = generate_tc(len_sec_v, fps)
        f.write('001  BL         AA   C         00:00:00:00 {} 00:00:00:00 {}\n'.format(tc_a_start, tc_a_start))
        f.write('002  {:10s} V    C         00:00:00:00 {} 00:00:00:00 {}\n'.format(sst_str + '_v', tc_v_len, tc_v_len))
        f.write('003  {:10s} AA   C         00:00:00:00 {} {} {}\n'.format(sst_str + '_a', tc_a_len, tc_a_start, tc_a_stop))

    f.close()