# Window
root = tk.Tk()
# Dimensions
root.geometry('200x100')
root.title('window')

def open_win():
    new = tk.Toplevel(root)
    new.geometry('200x100')
    new.title('new window')
    tk.Label(new, text='newWindow').pack(pady=30)
    new.mainloop()


# Button
btn_1 = tk.Button(root, text = 'Register User', command=open_win).pack()
btn_2 = tk.Button(root, text = 'Search match', command=open_win).pack()
# Position
#btn_1.pack(side='left')
#btn_2.pack(side="right")

root.mainloop()