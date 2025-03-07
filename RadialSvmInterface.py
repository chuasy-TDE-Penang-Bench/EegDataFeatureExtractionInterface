import re
import os
import subprocess
import threading
# import svmRbf
import svm
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

class EEGClassificationInterface:
    def __init__(self, root):

        # Constructor
        self.root = root
        self.root.title("EEG Classification Interface")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")

        # Instance Variables
        self.output_dir = tk.StringVar()
        self.train_data_path = tk.StringVar()
        self.test_data_path = tk.StringVar()
        self.model_file_path = tk.StringVar()
        self.progress_bar = None

        # GUI Setup
        self.create_widgets()

    def create_widgets(self):

        # Default attributes
        default_font = ("Arial", 10)
        log_font = ("Courier", 10)
        default_pad = 5
        button_bg = "#d9d9d9"
        frame_bg = "#f0f0f0"
        button_width = 12

        # Helper method to create buttons with default attributes
        def create_button(parent, text, command, width=button_width, font=default_font, bg=button_bg, side=tk.LEFT):
            button = tk.Button(parent, text=text, command=command, font=font, bg=bg, width=width)
            button.pack(side=side, padx=default_pad, pady=default_pad)
            return button

        # Output Directory Frame
        output_frame = tk.Frame(self.root, bg=frame_bg)
        output_frame.pack(pady=default_pad, fill=tk.X)

        # Output Directory Label
        label = tk.Label(output_frame, text="Output Directory:", bg=frame_bg, font=default_font)
        label.pack(side=tk.LEFT, padx=default_pad, pady=default_pad)

        # Output Directory Entry
        output_entry = tk.Entry(output_frame, textvariable=self.output_dir, font=default_font)
        output_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=default_pad, pady=default_pad)

        # Browse Button
        create_button(output_frame, "Browse", self.browse)

        # Train Data Frame
        train_data_frame = tk.Frame(self.root, bg=frame_bg)
        train_data_frame.pack(pady=default_pad, fill=tk.X)

        # Train Data Label
        label = tk.Label(train_data_frame, text="Train Data Path:", bg=frame_bg, font=default_font)
        label.pack(side=tk.LEFT, padx=default_pad, pady=default_pad)

        # Train Data Entry
        train_data_entry = tk.Entry(train_data_frame, textvariable=self.train_data_path, font=default_font)
        train_data_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=default_pad, pady=default_pad)

        # Browse Button
        create_button(train_data_frame, "Browse", self.browse_train_data)

        # Test Data Frame
        test_data_frame = tk.Frame(self.root, bg=frame_bg)
        test_data_frame.pack(pady=default_pad, fill=tk.X)

        # Test Data Label
        label = tk.Label(test_data_frame, text="Test Data Path:", bg=frame_bg, font=default_font)
        label.pack(side=tk.LEFT, padx=default_pad, pady=default_pad)

        # Test Data Entry
        test_data_entry = tk.Entry(test_data_frame, textvariable=self.test_data_path, font=default_font)
        test_data_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=default_pad, pady=default_pad)

        # Browse Button
        create_button(test_data_frame, "Browse", self.browse_test_data)

        # Model File Frame
        model_file_frame = tk.Frame(self.root, bg=frame_bg)
        model_file_frame.pack(pady=default_pad, fill=tk.X)

        # Model File Label
        label = tk.Label(model_file_frame, text="Model File Path:", bg=frame_bg, font=default_font)
        label.pack(side=tk.LEFT, padx=default_pad, pady=default_pad)

        # Model File Entry
        model_file_entry = tk.Entry(model_file_frame, textvariable=self.model_file_path, font=default_font)
        model_file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=default_pad, pady=default_pad)

        # Browse Button
        create_button(model_file_frame, "Browse", self.browse_model)

        # Button Frame
        button_frame = tk.Frame(self.root, bg=frame_bg)
        button_frame.pack(pady=default_pad, fill=tk.X)

        create_button(button_frame, "Start", self.start_svm, side=tk.RIGHT)

        # Progress Frame
        progress_frame = tk.Frame(self.root, bg=frame_bg)
        progress_frame.pack(pady=default_pad, fill=tk.X, anchor="w")

        # Progress Bar
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode="determinate", maximum=100)
        self.progress_bar.pack(side="left", fill=tk.X, expand=True, padx=default_pad, pady=default_pad)

        # Explore Button
        create_button(progress_frame, "Explore", self.show_output_directory, side=tk.RIGHT)

        # Log Frame
        log_frame = tk.Frame(self.root, bg=frame_bg)
        log_frame.pack(padx=default_pad, pady=default_pad, fill=tk.BOTH, expand=True)

        # Log Label
        log_label = tk.Label(log_frame, text="Log Output", font=default_font, bg=frame_bg)
        log_label.pack(anchor="center", padx=default_pad, pady=default_pad)

        # Log Box
        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD, bg="#F5F5F5", font=log_font)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # Clear Button Frame
        clear_button_frame = tk.Frame(log_frame, bg=frame_bg)
        clear_button_frame.pack(fill=tk.X, pady=default_pad)

        # Clear Log Button
        create_button(clear_button_frame, "Clear Log", self.clear_log, side=tk.RIGHT)

        # Exit Button Frame
        exit_button_frame = tk.Frame(self.root, bg=frame_bg)
        exit_button_frame.pack(padx=default_pad, pady=default_pad, fill=tk.X, side=tk.BOTTOM)

        # Exit Button
        create_button(exit_button_frame, "Exit", self.root.quit, side=tk.RIGHT)

    def browse(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
            self.log(f"Output directory set to: {os.path.normpath(directory)}")

    def browse_train_data(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.train_data_path.set(file)
            self.log(f"Train data path set to: {os.path.normpath(file)}")

    def browse_test_data(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.test_data_path.set(file)
            self.log(f"Test data path set to: {os.path.normpath(file)}")

    def browse_model(self):
        file = filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")])
        if file:
            self.model_file_path.set(file)
            self.log(f"Model file path set to: {os.path.normpath(file)}")

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.yview(tk.END)

    def start_svm(self):
        if not self.train_data_path:
            messagebox.showwarning("No Train Data Path", "Please select a train data path.")
            return
        if not self.test_data_path:
            messagebox.showwarning("No Test Data Path", "Please select a test data path.")
            return
        if not self.output_dir.get():
            messagebox.showwarning("No Output Directory", "Please select an output directory.")
            return
        if not self.model_file_path.get():
            response = messagebox.askyesno("No Model File Selected", "No model file is selected. Create a new model?")
            if not response:
                return

        target_process = svm.process

        if target_process:
            processing_thread = threading.Thread(target=target_process, args=(self,))
            processing_thread.start()

    def show_output_directory(self):
        output_dir = os.path.normpath(self.output_dir.get())
        subprocess.Popen(fr'explorer "{output_dir}"')

    def clear_log(self):
        self.log_box.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGClassificationInterface(root)
    root.mainloop()