import pandas as pd
from tkinter import filedialog as fd

# filename = fd.askopenfilename()
filename = r'E:\Work Roman\Git\blades\AtlasData.xlsx'
print(filename)
file = pd.ExcelFile(filename)
bladeData = pd.read_excel(filename,sheet_name=None,header=0)
# names,x,yss,yps = []
for key in bladeData:
    sheet = bladeData[key]
    print(sheet['X'])
    columns = sheet.columns
    print(columns)

print("DONE!")