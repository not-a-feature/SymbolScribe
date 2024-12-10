import tkinter as tk
import customtkinter as ctk
import onnxruntime as ort
from os import path
from PIL import Image, ImageDraw
from utils import crop_to_content, add_background_to_image
from symbols import symbols
import numpy as np

base_path = path.dirname(__file__)
model_path = path.abspath(path.join(base_path, "SymbolCNN.onnx"))
symbol_path = path.join(base_path, "symbols")
icon_path = path.join(base_path, "icon.ico")

image_size = (32, 32)  # Width / Height

about_text = f"""
SymbolScribe is a simple LaTeX symbol recognition tool.

It uses a Convolutional Neural Network (CNN) to predict the most likely LaTeX commands based on your sketch.

For more information, visit:
- juleskreuer.eu/projects/SymbolScribe
- github.com/not-a-feature/SymbolScribe/

SymbolScribe is build on Python.
License: PSF LICENSE AGREEMENT
Homepage: python.org


Following Libraries are used:

ONNX Runtime
License: MIT
Homepage: onnxruntime.ai

customtkinter
License: MIT
Homepage: customtkinter.tomschimansky.com

Pillow (PIL)
License: MIT-CMU
Homepage: python-pillow.org

NumPy
License: NumPy
Homepage: numpy.org

"""
# ONNX runtime session
ort_session = ort.InferenceSession(model_path)


def infer(input_img):
    cropped_img = crop_to_content(input_img, image_size)
    width, height = cropped_img.size

    # Resize and convert to grayscale
    resized_img = cropped_img.resize(image_size, Image.BILINEAR).convert("L")

    img_np = np.array(resized_img).astype(np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)  # Add channel dimension

    # ONNX runtime inference
    ort_inputs = {
        ort_session.get_inputs()[0].name: img_np,
        ort_session.get_inputs()[1].name: np.array([width], dtype=np.int64),
        ort_session.get_inputs()[2].name: np.array([height], dtype=np.int64),
    }
    output = ort_session.run(None, ort_inputs)[0]

    probabilities = np.exp(output) / np.sum(np.exp(output))  # Softmax
    top_indices = np.argsort(probabilities[0])[::-1][:5]  # Top 5 indices
    top_probs = probabilities[0][top_indices]

    results = []
    print("Infered Symbols:")
    for i in range(5):
        symbol_index = top_indices[i].item()
        probability = top_probs[i].item()
        symbol = symbols[symbol_index]
        results.append((symbol, probability))
        print(f"{symbol}: {probability:.4f}")
    print()
    return results


class SymbolScribeWindow:
    def __init__(self, main):
        self.main = main
        main.title("SymbolScribe")
        ctk.set_appearance_mode("System")  # Use system appearance mode
        ctk.set_default_color_theme("blue")
        main.after(201, lambda: root.iconbitmap(icon_path))

        self.line_width = 16

        self.canvas = tk.Canvas(
            main,
            bg="lightgray",
            width=500,
            height=300,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.Y)

        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Store lines for undo/reset
        self.lines = []
        self.current_line = None

        # Frame for buttons
        button_frame = ctk.CTkFrame(main, fg_color="transparent")
        button_frame.pack(pady=10)

        # Undo button
        self.undo_button = ctk.CTkButton(
            button_frame,
            text="Undo",
            command=self.undo_last_line,
            width=100,
        )
        self.undo_button.pack(side=tk.LEFT, padx=10)

        # Reset Button
        self.reset_button = ctk.CTkButton(
            button_frame,
            text="Reset",
            command=self.reset_canvas,
            width=100,
        )
        self.reset_button.pack(side=tk.LEFT, padx=10)

        # Output area (using CTkFrame and CTkLabels)
        output_frame = ctk.CTkFrame(main, fg_color="transparent")
        output_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        label_frame = ctk.CTkFrame(output_frame, cursor="hand2")  # Frame for each result
        label_frame.pack(pady=5, padx=10, fill=tk.X)

        ctk_bg_color = label_frame.cget("fg_color")

        self.bg_color_light = tuple((c // 256 for c in main.winfo_rgb(ctk_bg_color[0])))
        self.bg_color_dark = tuple((c // 256 for c in main.winfo_rgb(ctk_bg_color[1])))

        self.output_labels = []
        self.result = []

        light_img_path = path.abspath(path.join(symbol_path, "space.png"))
        dark_img_path = path.abspath(path.join(symbol_path, "space_dark.png"))

        light_img = Image.open(light_img_path)
        dark_img = Image.open(dark_img_path)

        # Create images with background color
        light_img = add_background_to_image(light_img, self.bg_color_light)
        dark_img = add_background_to_image(dark_img, self.bg_color_dark)

        ctk_img = ctk.CTkImage(
            light_image=light_img,
            dark_image=dark_img,
            size=(35, 35),
        )

        for i in range(5):
            if i:
                # Skip first frame generation as we already created one to get the background color.
                label_frame = ctk.CTkFrame(output_frame, cursor="hand2")  # Frame for each result
                label_frame.pack(pady=5, padx=10, fill=tk.X)

            symbol_img = ctk.CTkLabel(
                label_frame,
                image=ctk_img,
                text="",
                anchor="w",
            )
            symbol_img.pack(side=tk.LEFT, expand=False, padx=5)

            # Left Aligned Label
            symbol_label = ctk.CTkLabel(label_frame, text="", anchor="w")
            symbol_label.pack(side=tk.LEFT, fill=tk.X, padx=10, expand=True)

            # Right-aligned
            prob_label = ctk.CTkLabel(label_frame, text="", anchor="e")
            prob_label.pack(side=tk.RIGHT, padx=5)

            self.output_labels.append((symbol_img, symbol_label, prob_label))

    def draw_line(self, event):
        if self.current_line is None:
            x, y = event.x, event.y
            self.current_line = [(x, y)]
            self.line_id = self.canvas.create_line(
                x,
                y,
                x,
                y,
                fill="black",
                smooth=True,
                width=self.line_width,
            )

        else:
            x, y = event.x, event.y
            self.current_line.append((x, y))
            self.canvas.coords(
                self.line_id,
                *[coord for point in self.current_line for coord in point],
            )

    def on_mouse_up(self, event):
        if self.current_line:
            self.lines.append(self.current_line)
            self.current_line = None
            self.update_output()

    def undo_last_line(self):
        if self.lines:
            self.canvas.delete(self.line_id)
            self.lines.pop()

            if self.lines:
                self.update_output()
                self.line_id = self.canvas.find_all()[-1]
            else:
                self.update_output(reset=True)

    def reset_canvas(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)
        self.lines = []
        self.current_line = None
        self.update_output(reset=True)

    def update_output(self, reset=False):
        if reset:
            result = [(("", "space"), 0)] * 5
        else:
            img = Image.new("L", (500, 300), "white")
            draw = ImageDraw.Draw(img)

            for line in self.lines:
                draw.line(line, fill="black", width=self.line_width)

            result = infer(img)
            self.result = result

        for i, ((latex_cmd, filename), proba) in enumerate(result):
            symbol_img, symbol_label, prob_label = self.output_labels[i]

            # Clear previous widgets in symbol_label
            for widget in symbol_img.winfo_children():
                widget.destroy()  # Crucial: Remove old widgets

            try:
                light_img_path = path.abspath(path.join(symbol_path, f"{filename}.png"))
                dark_img_path = path.abspath(path.join(symbol_path, f"{filename}_dark.png"))

                # Open images using Pillow
                light_img = Image.open(light_img_path)
                dark_img = Image.open(dark_img_path)

                # Create images with background color
                light_img = add_background_to_image(light_img, self.bg_color_light)
                dark_img = add_background_to_image(dark_img, self.bg_color_dark)

                ctk_img = ctk.CTkImage(
                    light_image=light_img,
                    dark_image=dark_img,
                    size=(35, 35),
                )

                image_label = ctk.CTkLabel(symbol_img, image=ctk_img, text="")
                image_label.grid(row=i, column=0, padx=(0, 0))

            except FileNotFoundError:
                print(f"Warning: Image file not found: {filename}")

            symbol_label.configure(text=latex_cmd + "  ")  # Set text AFTER image
            symbol_label.bind("<Button-1>", lambda event, i=i: self.copy_latex(i))
            if proba:
                prob_label.configure(text=f"{proba*100:.2f}%")
            else:
                prob_label.configure(text="")

    def copy_latex(self, index):
        latex_cmd, _ = self.result[index][0]
        self.main.clipboard_clear()
        self.main.clipboard_append(latex_cmd)

        print(f"Copied LaTeX command: {latex_cmd}")

    def show_about(self):
        about_window = ctk.CTkToplevel(self.main)
        about_window.title("About SymbolScribe")

        about_label = ctk.CTkLabel(about_window, text=about_text, justify=tk.LEFT, wraplength=400)
        about_label.pack(padx=20, pady=20)

        close_button = ctk.CTkButton(about_window, text="Close", command=about_window.destroy)
        close_button.pack(pady=10)


root = ctk.CTk()
app = SymbolScribeWindow(root)


# Footer Frame (to hold both label and button)
footer_frame = ctk.CTkFrame(root, fg_color="transparent")
footer_frame.pack(side=tk.BOTTOM, fill=tk.X)


# Info Button
info_button = ctk.CTkButton(footer_frame, text="❓", width=20, height=20, command=app.show_about)
info_button.pack(side=tk.RIGHT, padx=5, pady=5)  # Add some padding

# Copyright Label
footer_label = ctk.CTkLabel(footer_frame, text="© Jules Kreuer 2025")
footer_label.pack(side=tk.RIGHT, padx=5)

root.mainloop()
