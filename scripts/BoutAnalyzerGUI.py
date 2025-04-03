import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import pandas as pd
import os

class BoutAnalyzerGUI(tk.Toplevel): # Changed from tk.Tk to tk.Toplevel
    def __init__(self, master=None): # Added master argument
        super().__init__(master) # Call Toplevel's init with master
        self.title("Behavior-Centric Video Bout Analyzer")
        self.geometry("950x700")
        self.minsize(800, 600)

        self.video_path = ""
        self.csv_path = ""
        self.cap = None
        self.behavior_bout_data = {}
        self.current_behavior = None
        self.current_bout_index = 0
        self.current_frame_index_in_bout = 0
        self.bout_status = {}
        self.accuracy_stats = {}
        self.playing = False
        self.playback_speed = 1.0
        self.after_id = None

        self.init_ui()

    def init_ui(self):
        # --- Configure Grid Layout ---
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- File Loading Frame ---
        file_frame = tk.Frame(self)
        file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.video_label = tk.Label(file_frame, text="Video File: Not Loaded")
        self.video_label.pack(side=tk.LEFT, padx=5)
        video_button = tk.Button(file_frame, text="Load Video", command=self.load_video_file)
        video_button.pack(side=tk.LEFT)

        self.csv_label = tk.Label(file_frame, text="CSV File: Not Loaded")
        self.csv_label.pack(side=tk.LEFT, padx=5)
        csv_button = tk.Button(file_frame, text="Load Labels CSV", command=self.load_csv_file)
        csv_button.pack(side=tk.LEFT)

        # --- Behavior and Bout Selection Frame ---
        select_frame = tk.Frame(self)
        select_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        behavior_select_frame = tk.Frame(select_frame)
        behavior_select_frame.pack(side=tk.LEFT, padx=10)
        behavior_select_label = tk.Label(behavior_select_frame, text="Select Behavior:")
        behavior_select_label.pack(side=tk.LEFT, padx=5)
        self.behavior_var = tk.StringVar()
        self.behavior_combobox = ttk.Combobox(behavior_select_frame, textvariable=self.behavior_var, state="readonly")
        self.behavior_combobox.pack(side=tk.LEFT)
        self.behavior_combobox.bind("<<ComboboxSelected>>", self.set_current_behavior_from_dropdown)

        bout_select_frame = tk.Frame(select_frame)
        bout_select_frame.pack(side=tk.LEFT, padx=10)
        bout_select_label = tk.Label(bout_select_frame, text="Select Bout ID:")
        bout_select_label.pack(side=tk.LEFT, padx=5)
        self.bout_id_var = tk.StringVar()
        self.bout_id_combobox = ttk.Combobox(bout_select_frame, textvariable=self.bout_id_var, state="readonly")
        self.bout_id_combobox.pack(side=tk.LEFT)
        self.bout_id_combobox.bind("<<ComboboxSelected>>", self.set_current_bout_from_dropdown)

        # --- Video Display Frame ---
        self.video_display_frame = tk.Frame(self, relief=tk.SOLID, borderwidth=1)
        self.video_display_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label_widget = tk.Label(self.video_display_frame)
        self.video_label_widget.pack(expand=tk.YES, fill=tk.BOTH)

        # --- Bout Info Frame ---
        bout_info_frame = tk.Frame(self)
        bout_info_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.behavior_display_label = tk.Label(bout_info_frame, text="Behavior: -")
        self.behavior_display_label.pack(side=tk.LEFT, padx=10)
        self.bout_id_label = tk.Label(bout_info_frame, text="Bout ID: -")
        self.bout_id_label.pack(side=tk.LEFT, padx=10)
        self.bout_frame_index_label_gui = tk.Label(bout_info_frame, text="Frame in Bout: -")
        self.bout_frame_index_label_gui.pack(side=tk.LEFT)

        # --- Navigation and Playback Frame ---
        self.nav_playback_frame = tk.Frame(self)
        self.nav_playback_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        prev_bout_button = tk.Button(self.nav_playback_frame, text="Previous Bout", command=self.prev_bout, state=tk.DISABLED)
        prev_bout_button.pack(side=tk.LEFT, padx=5)
        next_bout_button = tk.Button(self.nav_playback_frame, text="Next Bout", command=self.next_bout, state=tk.DISABLED)
        next_bout_button.pack(side=tk.LEFT, padx=5)
        prev_frame_button = tk.Button(self.nav_playback_frame, text="< Frame", command=self.previous_frame, state=tk.DISABLED)
        prev_frame_button.pack(side=tk.LEFT, padx=5)
        next_frame_button = tk.Button(self.nav_playback_frame, text="Frame >", command=self.next_frame, state=tk.DISABLED)
        next_frame_button.pack(side=tk.LEFT, padx=5)
        self.buttons_frame_nav = [prev_frame_button, next_frame_button]
        self.play_pause_button = tk.Button(self.nav_playback_frame, text="Play", command=self.play_pause_video, state=tk.DISABLED)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.StringVar(value="1x")
        speed_combobox = ttk.Combobox(self.nav_playback_frame, textvariable=self.speed_var, values=["0.5x", "1x", "2x"], state="readonly")
        speed_combobox.pack(side=tk.LEFT, padx=5)
        speed_combobox.bind("<<ComboboxSelected>>", self.set_playback_speed)

        # --- Confirmation Frame ---
        confirm_frame = tk.Frame(self)
        confirm_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        correct_button = tk.Button(confirm_frame, text="Correct", command=lambda: self.confirm_bout('correct'), state=tk.DISABLED)
        correct_button.pack(side=tk.LEFT, padx=5)
        incorrect_button = tk.Button(confirm_frame, text="Incorrect", command=lambda: self.confirm_bout('incorrect'), state=tk.DISABLED)
        incorrect_button.pack(side=tk.LEFT)
        self.buttons_confirm_bout = [correct_button, incorrect_button]

        # --- Accuracy Stats and Export Frame ---
        stats_export_frame = tk.Frame(self)
        stats_export_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=10)

        stats_frame = tk.Frame(stats_export_frame)
        stats_frame.pack(side=tk.LEFT, padx=10)
        self.total_bouts_label = tk.Label(stats_frame, text="Total Bouts: 0")
        self.total_bouts_label.pack(side=tk.TOP, anchor="w")
        self.checked_bouts_label = tk.Label(stats_frame, text="Bouts Checked: 0")
        self.checked_bouts_label.pack(side=tk.TOP, anchor="w")
        self.correct_bouts_label = tk.Label(stats_frame, text="Correct Bouts: 0")
        self.correct_bouts_label.pack(side=tk.TOP, anchor="w")
        self.incorrect_bouts_label = tk.Label(stats_frame, text="Incorrect Bouts: 0")
        self.incorrect_bouts_label.pack(side=tk.TOP, anchor="w")

        export_frame = tk.Frame(stats_export_frame)
        export_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        export_stats_button = tk.Button(export_frame, text="Export Stats", command=self.export_stats_to_csv)
        export_stats_button.pack(side=tk.TOP, anchor="w", fill=tk.X)
        export_bout_status_button = tk.Button(export_frame, text="Export Bout Status", command=self.export_bout_status_to_csv)
        export_bout_status_button.pack(side=tk.TOP, anchor="w", fill=tk.X)


    def load_video_file(self):
        file_path = filedialog.askopenfilename(
            master=self, # Pass self (Toplevel) as master
            defaultextension=".mp4",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.wmv;*.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"Video File: {os.path.basename(self.video_path)}")
            if self.csv_path:
                self.initialize_bouts()

    def load_csv_file(self):
        file_path = filedialog.askopenfilename(
            master=self, # Pass self (Toplevel) as master
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.csv_path = file_path
            self.csv_label.config(text=f"CSV File: {os.path.basename(self.csv_path)}")
            if self.video_path:
                self.initialize_bouts()

    def initialize_bouts(self):
        """Loads bout data from CSV, populates dropdowns, initializes video."""
        try:
            df = pd.read_csv(self.csv_path)
            grouped_by_behavior = df.groupby('Class Label')
            behaviors = sorted(list(grouped_by_behavior.groups.keys()))
            self.behavior_bout_data = {}
            self.accuracy_stats = {}

            overall_total_bouts = 0
            for behavior in behaviors:
                behavior_group = grouped_by_behavior.get_group(behavior)
                grouped_by_bout_id = behavior_group.groupby('Bout ID')
                bout_ids_for_behavior = sorted(list(grouped_by_bout_id.groups.keys()))
                self.behavior_bout_data[behavior] = {}
                behavior_total_bouts = 0
                for bout_id in bout_ids_for_behavior:
                    bout_group_data = grouped_by_bout_id.get_group(bout_id)
                    self.behavior_bout_data[behavior][bout_id] = bout_group_data
                    behavior_total_bouts += 1
                overall_total_bouts += behavior_total_bouts

                self.accuracy_stats[behavior] = {
                    'total_bouts': behavior_total_bouts,
                    'bouts_checked': 0,
                    'bouts_correct': 0,
                    'bouts_incorrect': 0
                }

            if not behaviors:
                messagebox.showerror("Error", "No behaviors found in CSV.", master=self) # Pass master
                return

            self.total_bouts_label.config(text=f"Total Bouts: {overall_total_bouts}")
            self.bout_status = {bout_id: 'unchecked' for behavior_data in self.behavior_bout_data.values() for bout_id in behavior_data.keys()}
            self.behavior_combobox['values'] = behaviors
            if behaviors:
                self.behavior_var.set(behaviors[0])

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open video file: {self.video_path}", master=self) # Pass master
                self.cap = None
                return

            if behaviors:
                self.set_current_behavior_from_dropdown()

            self.enable_navigation_and_confirmation()

        except FileNotFoundError:
            messagebox.showerror("Error", "CSV file not found.", master=self) # Pass master
        except KeyError:
            messagebox.showerror("Error", "CSV must contain 'Bout ID', 'Frame Number', 'Class Label'.", master=self) # Pass master
        except Exception as e:
            messagebox.showerror("Error", f"Error processing CSV: {e}", master=self) # Pass master
            self.behavior_bout_data = {}
            self.cap = None

    def enable_navigation_and_confirmation(self):
        """Enables navigation and confirmation buttons."""
        nav_buttons = [button for button in self.nav_playback_frame.winfo_children() if isinstance(button, tk.Button)]
        for button in nav_buttons + self.buttons_confirm_bout + self.buttons_frame_nav + [self.play_pause_button]:
            button.config(state=tk.NORMAL)
        self.behavior_combobox.config(state=tk.NORMAL)
        self.bout_id_combobox.config(state=tk.NORMAL)

    def set_current_behavior_from_dropdown(self, event=None):
        """Sets current behavior from dropdown, updates bout IDs."""
        selected_behavior = self.behavior_var.get()
        if selected_behavior in self.behavior_bout_data:
            self.current_behavior = selected_behavior
            bout_ids_for_behavior = sorted(list(self.behavior_bout_data[selected_behavior].keys()))
            self.bout_id_combobox['values'] = bout_ids_for_behavior
            if bout_ids_for_behavior:
                self.bout_id_var.set(bout_ids_for_behavior[0])
                self.current_bout_index = 0
                self.set_current_bout_from_dropdown()
            else:
                self.bout_id_combobox['values'] = []
                self.bout_id_var.set("")
                self.clear_video_display()
                self.bout_id_label.config(text="Bout ID: -")
                self.behavior_display_label.config(text="Behavior: -")
                self.bout_frame_index_label_gui.config(text="Frame in Bout: -")
        else:
            messagebox.showerror("Error", f"Invalid Behavior: '{selected_behavior}'.", master=self) # Pass master

    def set_current_bout_from_dropdown(self, event=None):
        """Sets current bout from dropdown, displays it."""
        selected_bout_id = self.bout_id_var.get()
        if not self.current_behavior:
            print("Error: current_behavior is None in set_current_bout_from_dropdown")
            return

        if not selected_bout_id:
            return

        try:
            bout_id = int(selected_bout_id)
        except ValueError:
            messagebox.showerror("Error", f"Invalid Bout ID format: '{selected_bout_id}'.", master=self) # Pass master
            return

        if self.current_behavior not in self.behavior_bout_data:
            messagebox.showerror("Error", f"Behavior '{self.current_behavior}' data error.", master=self) # Pass master
            return

        if bout_id in self.behavior_bout_data[self.current_behavior]:
            self.stop_playback()
            bout_ids_for_behavior = sorted(list(self.behavior_bout_data[self.current_behavior].keys()))
            self.current_bout_index = bout_ids_for_behavior.index(bout_id)
            self.current_frame_index_in_bout = 0
            self.show_bout()
        else:
            messagebox.showerror("Error", f"Invalid Bout ID '{selected_bout_id}' for '{self.current_behavior}'.", master=self) # Pass master

    def show_bout(self):
        """Displays the current bout frame with behavior label overlay (PIL)."""
        if not self.current_behavior: return
        bout_id_str = self.bout_id_var.get()
        if not bout_id_str: return
        bout_id = int(bout_id_str)

        if not (self.current_behavior in self.behavior_bout_data and bout_id in self.behavior_bout_data[self.current_behavior]):
            self.clear_video_display()
            self.bout_id_label.config(text="Bout ID: -")
            self.behavior_display_label.config(text="Behavior: -")
            self.bout_frame_index_label_gui.config(text="Frame in Bout: -")
            return

        bout_group = self.behavior_bout_data[self.current_behavior][bout_id]
        frame_numbers = bout_group['Frame Number'].tolist()

        if not frame_numbers:
            messagebox.showerror("Error", f"No frames for Behavior: {self.current_behavior}, Bout ID: {bout_id}", master=self) # Pass master
            return

        self.current_frame_index_in_bout = max(0, min(self.current_frame_index_in_bout, len(frame_numbers) - 1))
        frame_index_to_show = frame_numbers[self.current_frame_index_in_bout] - 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_to_show)
        ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # --- Draw Behavior Label on PIL Image ---
            draw = ImageDraw.Draw(img_pil)
            font = None
            try:
                font = ImageFont.truetype("arialbd.ttf", 40)  # Larger font, try Arial Bold again
            except IOError:
                font = ImageFont.load_default()

            if font:
                text = self.current_behavior  # Use the actual behavior name
                text_color = (0, 255, 0)  # Green color
                text_position = (50, 50)  # More central position
                draw.text(text_position, text, font=font, fill=text_color)
            # No else block needed now, as default font will load if Arial fails

            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label_widget.config(image=img_tk)
            self.video_label_widget.image = img_tk
            self.behavior_display_label.config(text=f"Behavior: {self.current_behavior}")
            self.bout_id_label.config(text=f"Bout ID: {bout_id}")
            self.bout_frame_index_label_gui.config(text=f"Frame in Bout: {self.current_frame_index_in_bout + 1}/{len(frame_numbers)}")

            prev_bout_button = [b for b in self.nav_playback_frame.winfo_children() if isinstance(b, tk.Button) and b.cget("text") == "Previous Bout"][0]
            next_bout_button = [b for b in self.nav_playback_frame.winfo_children() if isinstance(b, tk.Button) and b.cget("text") == "Next Bout"][0]
            prev_frame_button = self.buttons_frame_nav[0]
            next_frame_button = self.buttons_frame_nav[1]

            prev_bout_button.config(state=tk.NORMAL if self.current_bout_index > 0 else tk.DISABLED)
            bout_ids_for_behavior = sorted(list(self.behavior_bout_data[self.current_behavior].keys()))
            next_bout_button.config(state=tk.NORMAL if self.current_bout_index < len(bout_ids_for_behavior) - 1 else tk.DISABLED)
            prev_frame_button.config(state=tk.NORMAL if self.current_frame_index_in_bout > 0 else tk.DISABLED)
            next_frame_button.config(state=tk.NORMAL if self.current_frame_index_in_bout < len(frame_numbers) - 1 else tk.DISABLED)

        else:
            messagebox.showerror("Error", f"Error reading frame {frame_index_to_show + 1} from video.", master=self) # Pass master
            self.stop_playback()
    def next_bout(self):
        """Moves to the next bout."""
        self.stop_playback()
        if not self.current_behavior: return
        bout_ids_for_behavior = sorted(list(self.behavior_bout_data[self.current_behavior].keys()))
        if self.current_bout_index < len(bout_ids_for_behavior) - 1:
            self.current_bout_index += 1
            next_bout_id = bout_ids_for_behavior[self.current_bout_index]
            self.bout_id_var.set(str(next_bout_id))
            self.current_frame_index_in_bout = 0
            self.show_bout()
        else:
            messagebox.showinfo("Info", "Last bout for this behavior.", master=self) # Pass master

    def prev_bout(self):
        """Moves to the previous bout."""
        self.stop_playback()
        if not self.current_behavior: return
        if self.current_bout_index > 0:
            self.current_bout_index -= 1
            bout_ids_for_behavior = sorted(list(self.behavior_bout_data[self.current_behavior].keys()))
            prev_bout_id = bout_ids_for_behavior[self.current_bout_index]
            self.bout_id_var.set(str(prev_bout_id))
            self.current_frame_index_in_bout = 0
            self.show_bout()
        else:
            messagebox.showinfo("Info", "First bout for this behavior.", master=self) # Pass master

    def next_frame(self):
        """Moves to the next frame in bout."""
        self.stop_playback()
        bout_id_str = self.bout_id_var.get()
        if not bout_id_str or not self.current_behavior: return
        bout_id = int(bout_id_str)
        bout_group = self.behavior_bout_data[self.current_behavior][bout_id]
        frame_numbers = bout_group['Frame Number'].tolist()
        if self.current_frame_index_in_bout < len(frame_numbers) - 1:
            self.current_frame_index_in_bout += 1
            self.show_bout()

    def previous_frame(self):
        """Moves to the previous frame in bout."""
        self.stop_playback()
        if self.current_frame_index_in_bout > 0:
            self.current_frame_index_in_bout -= 1
            self.show_bout()

    def play_pause_video(self):
        """Toggles play/pause."""
        if not self.playing:
            self.playing = True
            self.play_pause_button.config(text="Pause")
            self.play_bout_frames()
        else:
            self.stop_playback()

    def stop_playback(self):
        """Stops playback."""
        if self.playing:
            self.playing = False
            self.play_pause_button.config(text="Play")
            if self.after_id:
                self.after_cancel(self.after_id)
                self.after_id = None

    def play_bout_frames(self):
        """Plays bout frames in a loop."""
        if not self.playing or not self.current_behavior: return

        bout_id_str = self.bout_id_var.get()
        if not bout_id_str: return
        bout_id = int(bout_id_str)
        bout_group = self.behavior_bout_data[self.current_behavior][bout_id]
        frame_numbers = bout_group['Frame Number'].tolist()

        if self.current_frame_index_in_bout < len(frame_numbers):
            self.show_bout()
            self.current_frame_index_in_bout += 1
            delay_ms = int(1000 / (self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed))
            self.after_id = self.after(delay_ms, self.play_bout_frames)
        else:
            self.stop_playback()
            self.current_frame_index_in_bout = 0
            self.show_bout() # Replay from start of bout

    def set_playback_speed(self, event=None):
        """Sets playback speed from dropdown."""
        speed_str = self.speed_var.get()
        if speed_str == "0.5x":
            self.playback_speed = 0.5
        elif speed_str == "1x":
            self.playback_speed = 1.0
        elif speed_str == "2x":
            self.playback_speed = 2.0

    def clear_video_display(self):
        """Clears video display to blank gray image."""
        blank_image = Image.new("RGB", (300, 200), "gray")
        blank_img_tk = ImageTk.PhotoImage(image=blank_image)
        self.video_label_widget.config(image=blank_img_tk)
        self.video_label_widget.image = blank_img_tk

    def export_stats_to_csv(self):
        """Exports accuracy stats to CSV, now per behavior."""
        filepath = filedialog.asksaveasfilename(
            master=self, # Pass self (Toplevel) as master
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Accuracy Statistics CSV"
        )
        if filepath:
            stats_list = []
            for behavior, stats in self.accuracy_stats.items():
                stats_row = {'Behavior': behavior}
                stats_row.update(stats)
                stats_list.append(stats_row)
            stats_df = pd.DataFrame(stats_list)
            try:
                stats_df.to_csv(filepath, index=False)
                messagebox.showinfo("Export Successful", f"Stats exported to:\n{filepath}", master=self) # Pass master
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting stats to CSV:\n{e}", master=self) # Pass master

    def export_bout_status_to_csv(self):
        """Exports bout status to CSV."""
        filepath = filedialog.asksaveasfilename(
            master=self, # Pass self (Toplevel) as master
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Bout Status CSV"
        )
        if filepath:
            bout_status_list = []
            for behavior, bout_data in self.behavior_bout_data.items():
                for bout_id in bout_data.keys():
                    status = self.bout_status.get(bout_id, 'unknown')
                    bout_status_list.append({'Behavior': behavior, 'Bout ID': bout_id, 'Status': status})
            status_df = pd.DataFrame(bout_status_list)
            try:
                status_df.to_csv(filepath, index=False)
                messagebox.showinfo("Export Successful", f"Bout status exported to:\n{filepath}", master=self) # Pass master
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting bout status to CSV:\n{e}", master=self) # Pass master

    def confirm_bout(self, status):
        """Confirms bout correctness, updates stats."""
        bout_id_str = self.bout_id_var.get()
        if not bout_id_str: return
        bout_id = int(bout_id_str)

        behavior_stats = self.accuracy_stats[self.current_behavior]

        if self.bout_status[bout_id] == 'unchecked':
            behavior_stats['bouts_checked'] += 1
        self.bout_status[bout_id] = status
        if status == 'correct':
            behavior_stats['bouts_correct'] += 1
        elif status == 'incorrect':
            behavior_stats['bouts_incorrect'] += 1

        self.update_accuracy_display()
        self.next_bout()

    def update_accuracy_display(self):
        """Updates accuracy stats labels in GUI."""
        current_behavior_stats = self.accuracy_stats.get(self.current_behavior, {'bouts_checked': 0, 'bouts_correct': 0, 'bouts_incorrect': 0})

        total_checked = sum(self.accuracy_stats[beh]['bouts_checked'] for beh in self.accuracy_stats)
        total_correct = sum(self.accuracy_stats[beh]['bouts_correct'] for beh in self.accuracy_stats)
        total_incorrect = sum(self.accuracy_stats[beh]['bouts_incorrect'] for beh in self.accuracy_stats)

        self.checked_bouts_label.config(text=f"Bouts Checked: {total_checked}")
        self.correct_bouts_label.config(text=f"Correct Bouts: {total_correct}")
        self.incorrect_bouts_label.config(text=f"Incorrect Bouts: {total_incorrect}")


if __name__ == "__main__":
    pass # Removed app.init_ui() and app.mainloop()