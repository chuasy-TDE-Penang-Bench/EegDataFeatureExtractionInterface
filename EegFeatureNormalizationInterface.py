import re
import os
import subprocess
import threading
import normalizeNormal
import normalizeStroke
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

class EEGFeatureNormalizationInterface:
    def __init__(self, root):

        # Constructor
        self.root = root
        self.root.title("EEG Feature Normalization Interface")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")

        # Instance Variables
        self.file_list = []
        self.output_dir = tk.StringVar()
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

        # File Table Frame
        file_frame = tk.Frame(self.root, bg=frame_bg)
        file_frame.pack(padx=default_pad, pady=default_pad, fill=tk.BOTH, expand=True)

        # File table with vertical scrollbar
        self.file_table = ttk.Treeview(file_frame, columns=("#1"), show="headings", height=6)
        self.file_table.heading("#1", text="Selected Files for Processing")
        self.file_table.column("#1", width=500)

        # Scrollbar setup
        file_scrollbar = tk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_table.yview)
        self.file_table.configure(yscrollcommand=file_scrollbar.set)

        # Pack widgets
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Button Frame
        button_frame = tk.Frame(self.root, bg=frame_bg)
        button_frame.pack(pady=default_pad, fill=tk.X)

        # Add Directory, Add File, Remove File, Reset and Start Button
        create_button(button_frame, "Add Directory", self.add_directory)
        create_button(button_frame, "Add File", self.add_file)
        create_button(button_frame, "Remove File", self.remove_file)
        create_button(button_frame, "Reset", self.reset)
        create_button(button_frame, "Start", self.start_normalization, side=tk.RIGHT)

        # Dropdown (Normal/Stroke)
        self.selection_var = tk.StringVar(value="Normal")
        selection_dropdown = tk.OptionMenu(button_frame, self.selection_var, "Normal", "Stroke")
        selection_dropdown.config(width=button_width, bg=button_bg)
        selection_dropdown.pack(side=tk.RIGHT, padx=default_pad, pady=default_pad)

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

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.yview(tk.END)

    # Helper method to add csv to file list and file table, and log the number of files added
    def add_csv(self, files):
        pattern = re.compile(r"P\d+.csv")
        selected_value = self.selection_var.get()

        for file in files:
            file_valid = ("Normal" in selected_value and "psd_summary.csv" in file) or \
                         ("Stroke" in selected_value and pattern.search(file))

            if file_valid:
                self.file_list.append(os.path.normpath(file))
                self.file_table.insert("", tk.END, values=(os.path.normpath(file),))
                self.log(f"Added file: {os.path.normpath(file)}")

    def add_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            files = [os.path.join(root, file) for root, _, file_list in os.walk(directory) for file in file_list]
            self.add_csv(files)

    def add_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file:
            self.add_csv([file])

    def remove_file(self):
        selected_items = self.file_table.selection()
        if selected_items:
            for item in selected_items:
                file_path = self.file_table.item(item, "values")[0]
                self.file_list.remove(file_path)
                self.file_table.delete(item)
                self.log(f"Removed file: {os.path.normpath(file_path)}")

    def reset(self):
        self.file_list = []
        for item in self.file_table.get_children():
            self.file_table.delete(item)
        self.progress_bar['value'] = 0
        self.log("Interface has been reset.")

    def start_normalization(self):
        if not self.file_list:
            messagebox.showwarning("No Files", "Please add files before starting.")
            return
        if not self.output_dir.get():
            messagebox.showwarning("No Output Directory", "Please select an output directory.")
            return

        target_process = None
        selected_value = self.selection_var.get()
        if "Normal" in selected_value:
            target_process = normalizeNormal.process

        elif "Stroke" in selected_value:
            target_process = normalizeStroke.process

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
    app = EEGFeatureNormalizationInterface(root)
    root.mainloop()