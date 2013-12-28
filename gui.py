import tkinter as tk
from tkinter.ttk import Progressbar
from tkinter import filedialog, messagebox
import os
import errno
from threading import Thread
import program_logic


def get_paths_in_dir(directory):
    filenames = os.listdir(directory)
    return [os.path.abspath(os.path.join(directory, name)) for name in filenames]


class MainFrame(tk.Frame):
    def __init__(self, parent):
        super(MainFrame, self).__init__(parent)
        self.parent = parent

        self.dir_var = tk.StringVar()
        self.audio_progress_var = tk.IntVar()
        self.video_progress_var = tk.IntVar()
        self.fps_var = tk.StringVar()
        self.start_button = None

        self.generate_ui()
        self.center_window()

    def generate_ui(self):
        self.parent.title('audiosync')

        dir_frame = tk.Frame(self)
        dir_frame.pack(fill=tk.X)
        tk.Label(dir_frame, text='Directory:').pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=self.dir_var).pack(fill=tk.X, expand=1, side=tk.LEFT)
        tk.Button(dir_frame, text='Select', command=self.select_dir).pack(side=tk.LEFT)
        tk.Button(dir_frame, text='Create Structure', command=self.create_directory_structure).pack(side=tk.RIGHT)

        fps_frame = tk.Frame(self)
        fps_frame.pack()
        tk.Label(fps_frame, text='FPS:').pack(side=tk.LEFT)
        tk.Entry(fps_frame, textvariable=self.fps_var).pack(side=tk.LEFT)

        cmd_frame = tk.Frame(self)
        cmd_frame.pack(fill=tk.X)
        self.start_button = tk.Button(cmd_frame, text='Start', command=self.execute)
        self.start_button.pack()

        Progressbar(self, variable=self.video_progress_var).pack(fill=tk.X)
        Progressbar(self, variable=self.audio_progress_var).pack(fill=tk.X)

        self.pack(fill=tk.BOTH, expand=1)

    def center_window(self):
        w = 500
        h = 120

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw - w) / 2
        y = (sh - h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def select_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path != '':
            self.dir_var.set(dir_path)

    def create_directory_structure(self):
        dir_path = self.dir_var.get()
        if dir_path != '':
            dir_names = ['video', 'audio', 'edl']
            for dir_name in dir_names:
                new_dir_path = os.path.join(dir_path, dir_name)
                try:
                    os.makedirs(new_dir_path)
                except OSError as ex:
                    if ex.errno != errno.EEXIST:
                        raise

    def execute(self):
        directory = self.dir_var.get()
        if directory is '':
            messagebox.showerror(title='audiosync', message='No directory selected')
            return

        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror(title='audiosync', message='FPS has to be decimal number')
            return

        thread = Thread(target=self.thread_target,
                        args=(self.audio_progress_var, self.video_progress_var, self.start_button, fps, directory))
        thread.start()
        self.start_button.config(state='disabled')


    @staticmethod
    def thread_target(audio_progress_var, video_progress_var, start_button, fps, directory):
        video_ret = analyse_directory(os.path.join(directory, 'video'), video_progress_var)
        audio_ret = analyse_directory(os.path.join(directory, 'audio'), audio_progress_var)

        program_logic.rename_files(audio_ret, 'a')
        program_logic.rename_files(video_ret, 'v')

        program_logic.generate_edls(video_ret, audio_ret, fps, os.path.join(directory, 'edl'))

        audio_progress_var.set(0)
        video_progress_var.set(0)

        start_button.config(state='normal')


def analyse_directory(directory, progress_var):
    ret_list = []
    files = os.listdir(directory)
    for n, filename in enumerate(files):
        path = os.path.abspath(os.path.join(directory, filename))
        result = program_logic.analyse_file(path)
        if result is not None:
            ret_list.append(result)
        progress_var.set(int((n + 1) / len(files) * 100))
    return ret_list


if __name__ == '__main__':
    root = tk.Tk()
    app = MainFrame(root)
    root.mainloop()