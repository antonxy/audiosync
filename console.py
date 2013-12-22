import argparse
import os
import program_logic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vd', help='Video directory')
    parser.add_argument('ad', help='Audio directory')
    parser.add_argument('edl', help='EDL output directory')
    parser.add_argument('fps', help='Project frames per second')
    args = parser.parse_args()

    audio_ret = analyse_directory(args.ad)
    video_ret = analyse_directory(args.vd)

    print('audio_ret = %s' % audio_ret)
    print('video_ret = %s' % video_ret)

    program_logic.rename_files(audio_ret, 'a')
    program_logic.rename_files(video_ret, 'v')

    fps = float(args.fps)

    program_logic.generate_edls(video_ret, audio_ret, fps, args.edl)


def analyse_directory(directory):
    ret_list = []
    for filename in os.listdir(directory):
        path = os.path.abspath(os.path.join(directory, filename))
        result = program_logic.analyse_file(path)
        if result is not None:
            ret_list.append(result)
    return ret_list

if __name__ == '__main__':
    main()