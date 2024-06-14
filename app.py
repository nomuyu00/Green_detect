import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression


class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("緑の物体捉える君")
        self.root.geometry("1200x800")  # ウィンドウのサイズを大きく設定

        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=20)

        self.canvas = None

    def upload_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4"), ("All files", "*.*")])
        if video_path:
            self.process_video(video_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        coordinates = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    coordinates.append((frame_number, int(x), int(y)))
                    break
        cap.release()
        self.calculate_distances_and_plot(coordinates)

    def calculate_distances_and_plot(self, coordinates):
        distances = []
        frames = []
        prev_coord = None
        for i, coord in enumerate(coordinates):
            frame, x, y = coord
            if prev_coord is not None and frame >= 5:
                prev_frame, prev_x, prev_y = prev_coord
                distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                distances.append(distance)
                frames.append(frame)
                if frame >= 15:
                    if len(distances) > 1:
                        mean_distance = np.mean(distances[:-1])
                        std_distance = np.std(distances[:-1])
                        if abs(distance - mean_distance) > 2 * std_distance:
                            messagebox.showinfo("Outlier Detected",
                                                f"Outlier detected at frame {frame} with distance {distance:.2f}")
                            distances.pop()
                            frames.pop()
                            break
            prev_coord = coord

        if len(frames) == 0 or len(distances) == 0:
            messagebox.showerror("Error", "No green object detected or insufficient data.")
        else:
            self.plot_graph(frames, distances)

    def plot_graph(self, frames, distances):
        mean_distance = np.mean(distances)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frames, distances, label='Distance')
        frames_np = np.array(frames).reshape(-1, 1)
        distances_np = np.array(distances).reshape(-1, 1)
        reg = LinearRegression().fit(frames_np, distances_np)
        line = reg.predict(frames_np)
        ax.plot(frames, line, color='red', linestyle='--', label='Linear Regression')

        # 平均の移動距離を青色の破線で表示
        mean_line = [mean_distance] * len(frames)
        ax.plot(frames, mean_line, color='blue', linestyle='--', label='Average Distance')

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Distance(pixel)')
        ax.set_title('Distance per Frame')
        ax.legend()
        ax.grid(True)

        # 平均の移動距離をグラフの上部に大きく表示
        plt.text(0.5, 1.08, f'Average Distance: {mean_distance:.2f}pixel', ha='center', va='center', transform=ax.transAxes,
                 fontsize=14, fontweight='bold')

        # 平均の移動速度を計算
        vertical_length = 0.24  # 映っている縦の長さ（ｍ）
        vertical_pixel = 736  # 縦のピクセル数
        frames_per_second = 40000  # 一秒あたりのフレーム数
        mean_velocity = mean_distance * vertical_length * frames_per_second / vertical_pixel

        # 平均の移動速度をグラフの上部に表示
        plt.text(0.5, 1.12, f'Average Velocity: {mean_velocity:.2f} m/s', ha='center', va='center', transform=ax.transAxes,
                 fontsize=14, fontweight='bold')

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()
