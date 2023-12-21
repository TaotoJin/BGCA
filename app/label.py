import tkinter as tk
import pandas as pd

# Load data
# df = pd.read_csv('your_data.csv') # Assuming your data is in a CSV file
test_data = {'text':[
    "Harumi Sushi has the freshest and most delicious array of sushi in NYC ."]
}
df = pd.DataFrame(test_data)

class LabelingApp:
    def __init__(self, master, data):
        self.master = master
        self.data = data
        self.current_index = 0
        self.selected_words = []

        self.label_var = tk.StringVar()
        self.text_frame = tk.Frame(master)
        self.text_frame.pack()

        self.setup_ui()

    def setup_ui(self):
        self.display_comment()
        tk.Button(self.master, text="Negative", command=lambda: self.set_label("negative")).pack()
        tk.Button(self.master, text="Next", command=self.next_comment).pack()

    def display_comment(self):
        # Clear previous text
        for widget in self.text_frame.winfo_children():
            widget.destroy()

        comment = self.data.iloc[self.current_index]
        comment = comment.text
        words = comment.split() # Assuming each comment is a string

        for word in words:
            word_btn = tk.Button(self.text_frame, text=word, command=lambda w=word: self.select_word(w))
            word_btn.pack(side=tk.LEFT)

    def select_word(self, word):
        if word not in self.selected_words:
            self.selected_words.append(word)
            # Update UI to reflect selected word

    def set_label(self, label):
        # Assign label to selected words
        # Update dataset
        pass

    def next_comment(self):
        self.current_index += 1
        self.selected_words = []
        self.display_comment()

# Main
root = tk.Tk()
app = LabelingApp(root, df)
root.mainloop()
