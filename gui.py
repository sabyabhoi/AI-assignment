import tkinter
import time
import customtkinter as ctk
import utils

# System settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.2)

# App frame
app = ctk.CTk()
app.geometry("720x480")
app.title("AI Assignment")

# Add UI elements
title = ctk.CTkLabel(app, text="Hyper-Parameter Optimization using AI Algorithms")
title.pack(padx=10, pady=10)

# Stock Input
stock_label = ctk.CTkLabel(app, text="Enter stock ticker")
stock_label.pack()

entry = ctk.CTkEntry(app, placeholder_text="MSFT")
entry.pack()

algo_label = ctk.CTkLabel(app, text="Select Algorithm")
algo_label.pack()


# Add Drop down menu
def combobox_callback(choice):
    print(f"{choice} selected")


values = ["Genetic Algorithm", "Simulated Annealing"]
combobox = ctk.CTkComboBox(app, width=200, values=values, command=combobox_callback)

combobox.set(values[1])
combobox.pack()

status_label = ctk.CTkLabel(app, text="")


def button_event():
    algo = combobox.get()
    if algo == "":
        algo = "MSFT"
    status_label.configure(text=f"Running {algo}")
    status_label.update()
    params = utils.run_algo(entry.get(), algo)
    status_label.configure(text=f"Completed with optimal parameters: {params}")
    status_label.update()


button = ctk.CTkButton(app, text="Optimize", command=button_event)
button.pack()

status_label.pack()

# Run app
app.mainloop()
