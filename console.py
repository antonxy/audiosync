import argparse
import os
import read_audio
import analyse_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vd', help='Video directory')
    parser.add_argument('ad', help='Audio directory')
    args = parser.parse_args()

    audio_ret = analyse_directory(args.ad)
    video_ret = analyse_directory(args.vd)

    print audio_ret, video_ret


def analyse_directory(directory):
    ret_list = []
    for filename in os.listdir(directory):
        path = os.path.abspath(os.path.join(directory, filename))

        sr, audio = read_audio.from_file_normalized(path)

        sync_point, data = analyse_audio.find_and_decode_signal(audio, sr, 4000, 0.05, 5000, 0.05)

        ret_list.append((path, sync_point, data))
    return ret_list

if __name__ == '__main__':
    main()