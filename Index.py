from tkinter import *
#import  model_building as mb
import os
root = Tk()
root.configure(background="white")
root.geometry("1120x420")
root.configure(background='#d7c0ae')
def function2():
    os.system("py data_analysis.py")
def function3():
    os.system("py datapreprocessing.py")
def function4():
    os.system("py modelbuilding.py")
def function5():
    os.system("py gui.py")
def function6():
    root.destroy()
root.title("Image Caption Generator")

label = Label(root, text="Image Caption Generator",
              font=("", 15),height=2,width=100)
label.grid(row=0,column=0,columnspan=2,padx=5, pady=5)

button = Button(root,text="Data Analysis",font=('', 15),width=35,height=3,command=function2)
button.grid(row=1,column=0,padx=5, pady=10)

button1 = Button(root,text="Data Preprocessing",font=('', 15),width=35,height=3,command=function3)
button1.grid(row=1,column=1,padx=5, pady=10)

button2 = Button(root,text="Model Building",font=('', 15),width=35,height=3,command=function4)
button2.grid(row=2,column=0,padx=5, pady=10)

button3 = Button(root,text="Image Caption Generator",font=('', 15),width=35,height=3,command=function5)
button3.grid(row=2,column=1,padx=5, pady=10)

button4 = Button(root,text="Exit",font=('', 15),width=35,height=3,command=function6)
button4.grid(row=3,column=0,columnspan=2,padx=5, pady=10)

root.mainloop()
