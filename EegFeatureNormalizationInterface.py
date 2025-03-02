import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import time


def process_psd_summary(input_dir, output_dir, is_train, progress_callback):
    if not os.path.exists(input_dir):
        return
    files = os.listdir(input_dir)
    total_files = len(files)

    for index, file in enumerate(files, start=1):
        time.sleep(0.1)  # Simulate processing time
        progress_callback(index, total_files)



class EEGDataFeatureExtractionInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Data Feature Extraction Interface")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")

        self.train_input_dir = tk.StringVar()
        self.test_input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.progress_bar = None
        self.create_widgets()

    def create_widgets(self):
        default_font = ("Arial", 10)
        log_font = ("Courier", 10)
        default_pad = 5
        button_bg = "#d9d9d9"
        frame_bg = "#f0f0f0"
        button_width = 12

        def create_button(parent, text, command, width=button_width, font=default_font, bg=button_bg, side=tk.LEFT):
            button = tk.Button(parent, text=text, command=command, font=font, bg=bg, width=width)
            button.pack(side=side, padx=default_pad, pady=default_pad)
            return button

        def create_directory_selector(label_text, variable):
            frame = tk.Frame(self.root, bg="#f0f0f0")
            frame.pack(pady=default_pad, fill=tk.X)

            label = tk.Label(frame, text=label_text, bg="#f0f0f0", font=default_font)
            label.pack(side=tk.LEFT, padx=default_pad, pady=default_pad)

            entry = tk.Entry(frame, textvariable=variable, font=default_font)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=default_pad, pady=default_pad)

            create_button(frame, "Browse", lambda: self.browse_directory(variable))
            return frame

        create_directory_selector("Train Input Directory:", self.train_input_dir)
        create_directory_selector("Test Input Directory:", self.test_input_dir)
        create_directory_selector("Output Directory:", self.output_dir)

        start_button_frame = tk.Frame(self.root, bg=frame_bg)
        start_button_frame.pack(fill=tk.X, pady=default_pad, anchor="e")
        tk.Button(start_button_frame, text="Start", command=self.start_normalization_threaded, font=default_font,
                  bg=button_bg, width=button_width).pack(side=tk.RIGHT, padx=default_pad)

        progress_frame = tk.Frame(self.root, bg=frame_bg)
        progress_frame.pack(pady=default_pad, fill=tk.X, anchor="w")

        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate", maximum=100)
        self.progress_bar.pack(side="left", fill=tk.X, expand=True, padx=default_pad, pady=default_pad)

        create_button(progress_frame, "Explore", self.show_output_directory, side=tk.RIGHT)

        log_frame = tk.Frame(self.root, bg=frame_bg)
        log_frame.pack(padx=default_pad, pady=default_pad, fill=tk.BOTH, expand=True)

        log_label = tk.Label(log_frame, text="Log Output", font=default_font, bg=frame_bg)
        log_label.pack(anchor="center", padx=default_pad, pady=default_pad)

        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD, bg="#F5F5F5", font=log_font)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        clear_button_frame = tk.Frame(log_frame, bg=frame_bg)
        clear_button_frame.pack(fill=tk.X, pady=default_pad)

        create_button(clear_button_frame, "Clear Log", self.clear_log, side=tk.RIGHT)

        exit_button_frame = tk.Frame(self.root, bg=frame_bg)
        exit_button_frame.pack(padx=default_pad, pady=default_pad, fill=tk.X, side=tk.BOTTOM)

        create_button(exit_button_frame, "Exit", self.root.quit, side=tk.RIGHT)

    def browse_directory(self, variable):
        directory = filedialog.askdirectory()
        if directory:
            variable.set(directory)
            self.log(f"Directory set: {os.path.normpath(directory)}")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.yview(tk.END)

    def show_output_directory(self):
        output_dir = os.path.normpath(self.output_dir.get())
        subprocess.Popen(fr'explorer "{output_dir}"')

    def clear_log(self):
        self.log_box.delete("1.0", tk.END)

    def start_normalization_threaded(self):
        threading.Thread(target=self.start_normalization, daemon=True).start()

    def start_normalization(self):
        train_dir = self.train_input_dir.get()
        test_dir = self.test_input_dir.get()
        output_dir = self.output_dir.get()

        if not train_dir or not test_dir or not output_dir:
            self.log("Error: Please set all directories before starting normalization.")
            return

        self.log("Starting normalization...")
        self.progress_bar['value'] = 0
        self.root.update_idletasks()

        total_files = 0
        if os.path.exists(train_dir):
            total_files += len(os.listdir(train_dir))
        if os.path.exists(test_dir):
            total_files += len(os.listdir(test_dir))

        if total_files == 0:
            self.log("No files found in the directories.")
            return

        def progress_callback(current_index, total):
            progress_percent = int((current_index / total) * 100)
            self.update_progress(progress_percent)

        process_psd_summary(train_dir, output_dir, is_train=True, progress_callback=progress_callback)
        process_psd_summary(test_dir, output_dir, is_train=False, progress_callback=progress_callback)

        self.update_progress(100)
        self.log("Normalization completed.")

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGDataFeatureExtractionInterface(root)
    root.mainloop()