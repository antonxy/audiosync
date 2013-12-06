import argparse
import os
import read_audio
import analyse_audio
from pytimecode import PyTimeCode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vd', help='Video directory')
    parser.add_argument('ad', help='Audio directory')
    parser.add_argument('edl', help='EDL output directory')
    parser.add_argument('fps', help='Project frames per second')
    args = parser.parse_args()

    audio_ret = analyse_directory(args.ad)
    video_ret = analyse_directory(args.vd)

    print 'audio_ret = %s' % audio_ret
    print 'video_ret = %s' % video_ret

    rename_files(audio_ret, 'a')
    rename_files(video_ret, 'v')

    generate_edls(video_ret, audio_ret, args.fps, args.edl)


def analyse_directory(directory):
    ret_list = []
    for filename in os.listdir(directory):
        path = os.path.abspath(os.path.join(directory, filename))

        try:
            sr, audio = read_audio.from_file_normalized(path)
        except Exception:
            print 'Could not open file %s' % path
            continue

        length = audio.size
        sync_point, data = analyse_audio.find_and_decode_signal(audio, sr, 4000, 0.05, 5000, 0.05)

        print 'path {} analysed'.format(path)

        ret_list.append({'path': path, 'sync_point_samples': sync_point,
                         'sst': data, 'sample_rate': sr, 'length_samples': length})
    return ret_list


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
                print v['sst']
                generate_edl(v, a, fps, edl_dir)


def generate_tc(seconds, fps):
    ptc = PyTimeCode(fps)
    ptc.float_to_tc(seconds)
    return ptc.make_timecode()


def generate_edl(video_data, audio_data, fps, edl_dir):
    sst = video_data['sst']
    sst_str = '-'.join(['%s' % e for e in sst])
    filename = sst_str + '.edl'

    f = open(os.path.join(os.path.abspath(edl_dir), filename), 'w')
    print 'generating {}'.format(filename)

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


if __name__ == '__main__':
    main()