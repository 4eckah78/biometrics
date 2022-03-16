from tkinter import *
from tkinter import filedialog
from template_method import template_matching
from Viola_Jones import detect_face

image_path = ""
template_path = ""


def browseFiles():
    global image_path
    image_path = filedialog.askopenfilename(initialdir="/",
                                            title="Select a File",
                                            filetypes=(("Image files",
                                                        "*.jpg*"),
                                                       ("all files",
                                                        "*.*")))
    label_file_explorer.configure(text="File Opened")


def browseFiles_template():
    global template_path
    template_path = filedialog.askopenfilename(initialdir="/",
                                               title="Select a File",
                                               filetypes=(("Image files",
                                                           "*.jpg*"),
                                                          ("all files",
                                                           "*.*")))

    label_file_explorer.configure(text="File Opened")


# Create the root window
window = Tk()

# Set window title
window.title('File Explorer')

# Set window size
window.geometry("500x500")

# Set window background color
window.config(background="white")

# Create a File Explorer label
label_file_explorer = Label(window,
                            text="GUI",
                            width=100, height=4,
                            fg="blue")

button_explore = Button(window,
                        text="Select Image",
                        command=browseFiles)

button_explore_2 = Button(window,
                          text="Select Template",
                          command=browseFiles_template)

button_detect_1 = Button(window,
                         text="Detect with template matching",
                         command=lambda: template_matching(image_path, template_path))

button_detect_2 = Button(window,
                         text="Detect with Viola-Jones",
                         command=lambda: detect_face(image_path))

button_exit = Button(window,
                     text="Exit",
                     command=exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column=1, row=1)

button_explore.grid(column=1, row=2)

button_exit.grid(column=1, row=6)

button_explore_2.grid(column=1, row=3)

button_detect_1.grid(column=1, row=4)

button_detect_2.grid(column=1, row=5)
# Let the window wait for any events
window.mainloop()
