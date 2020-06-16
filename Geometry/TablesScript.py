import xlrd
import os
import numpy as np

dirList = os.listdir(os.getcwd())
dirList.remove('.vscode')
dirList.remove('TablesScript.py')
print(dirList)
a = os.walk(os.getcwd())
print(a)

print('Success')