import xlrd
from tkinter import filedialog as fd

filename = fd.askopenfilename()
print(filename)
bladeData = xlrd.open_workbook(filename)
a = 5
print("DONE!")