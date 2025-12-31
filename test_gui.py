"""
Simple test to verify GUI can open
"""
import tkinter as tk

def test():
    root = tk.Tk()
    root.title("Test Window")
    root.geometry("400x300")
    
    label = tk.Label(root, text="If you see this, GUI is working!", font=('Arial', 14))
    label.pack(pady=50)
    
    button = tk.Button(root, text="Close", command=root.quit)
    button.pack()
    
    root.mainloop()
    print("GUI test completed")

if __name__ == "__main__":
    test()

