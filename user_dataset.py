import tkinter as tk
import tkinter.font as font
import os
from PIL import Image, ImageDraw, ImageTk  # Import ImageTk

from symbols import symbols
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use("TkAgg")


class SymbolScribeWindow:
    def __init__(self, master, start_index):
        self.master = master
        self.current_symbol_index = start_index
        self.symbols = symbols
        master.title("Drawing App")
        font.nametofont("TkDefaultFont").configure(size=32)

        self.line_width = 16  # Line width
        self.symbol_label = tk.Label(master, text=f"")
        self.symbol_label.pack(side=tk.LEFT, padx=20)  # Added padding

        # Canvas for drawing
        self.canvas = tk.Canvas(master, bg="white", width=500, height=300)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Store lines for undo/reset
        self.lines = []
        self.current_line = None

        # Frame for buttons
        button_frame = tk.Frame(master)
        button_frame.pack()

        # Undo button
        self.undo_button = tk.Button(
            button_frame,
            text="Undo",
            command=self.undo_last_line,
        )
        self.undo_button.pack(side=tk.LEFT, padx=5)

        # Reset Button
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_canvas)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Save & Next Button
        self.reset_button = tk.Button(button_frame, text="Save & Next", command=self.save_and_next)
        self.reset_button.pack(side=tk.LEFT, padx=5)

    def draw_line(self, event):
        if self.current_line is None:
            x, y = event.x, event.y
            self.current_line = [(x, y)]
            self.line_id = self.canvas.create_line(
                x, y, x, y, fill="black", smooth=True, width=self.line_width
            )  # Apply width

        else:
            x, y = event.x, event.y
            self.current_line.append((x, y))
            self.canvas.coords(
                self.line_id, *[coord for point in self.current_line for coord in point]
            )

    def on_mouse_up(self, event):
        if self.current_line:
            self.lines.append(self.current_line)  # Store the list of points
            self.current_line = None

    def undo_last_line(self):
        if self.lines:
            self.canvas.delete(self.line_id)  # Delete by line ID
            self.lines.pop()
            self.update_output()
            if self.lines:  # Update line_id if lines still exist
                self.line_id = self.canvas.find_all()[-1]  # Get the ID of the last drawn line

    def reset_canvas(self):
        for item in self.canvas.find_all():  # Delete all items on the canvas
            self.canvas.delete(item)
        self.lines = []
        self.current_line = None  # Important: reset current_line
        self.update_output()

    def render_latex(self, symbol):
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.text(
            0.5, 0.5, f"${symbol}$", fontsize=48, ha="center", va="center"
        )  # Increase font size
        ax.axis("off")
        fig.canvas.draw()

        img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)  # Close the plot to free resources

        return ImageTk.PhotoImage(img)

    def save_and_next(self):
        # Save the image
        img = Image.new("L", (500, 300), "white")
        draw = ImageDraw.Draw(img)
        for line in self.lines:
            draw.line(line, fill="black", width=self.line_width)

        cleaned_name = self.symbols[self.current_symbol_index][1]

        i = 0
        filename = os.path.join(output_dir, f"{cleaned_name}_drawn_{i}.png")
        while os.path.isfile(filename):
            i = i + 1
            filename = os.path.join(output_dir, f"{cleaned_name}_drawn_{i}.png")

        print(self.current_symbol_index, "Saved", filename)
        img.save(filename)

        with open(csv_file, "a") as f:
            f.write(f"{cleaned_name}_drawn_{i}.png,{self.symbols[self.current_symbol_index][0]}\n")

        # Clear canvas and go to next symbol
        self.canvas.delete("all")
        self.lines = []
        self.current_line = None

        self.current_symbol_index = (self.current_symbol_index + 1) % len(self.symbols)
        latex_image = self.render_latex(self.symbols[self.current_symbol_index][0])
        self.symbol_label.config(image=latex_image)
        self.symbol_label.image = latex_image  # keep a reference

    def undo_last_line(self):
        if self.lines:
            self.canvas.delete(self.line_id)
            self.lines.pop()
            if self.lines:
                self.line_id = self.canvas.find_all()[-1]

    def reset_canvas(self):
        self.canvas.delete("all")
        self.lines = []
        self.current_line = None


# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "test_dataset")
csv_file = os.path.join(base_dir, "test_dataset.csv")
image_size = (128, 64)

#  Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# start_index = random.randint(0, len(symbols) - 1)
start_index = 0
root = tk.Tk()
app = SymbolScribeWindow(root, start_index)
# Render the initial symbol
latex_image = app.render_latex(symbols[start_index][0])
app.symbol_label.config(image=latex_image)  # Use config with image
app.symbol_label.image = latex_image  # Keep a reference!


def ext_save(event):
    app.save_and_next()


root.bind("<space>", ext_save)


root.mainloop()
