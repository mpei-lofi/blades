import pandas as pd

def ReadBladeDB(filename, blade_name = 'All', array_format = False):
    file = pd.ExcelFile(filename)
    bladeData = pd.read_excel(filename,sheet_name=None,header=0)
    result = 'Здесь ничего нет, проверь все ещё раз'
    if not array_format:
        if blade_name == 'All':
            result = bladeData
        else:
            if blade_name in bladeData:
                result = bladeData[blade_name]
                print('Профиль {0} считан'.format(blade_name))
            else:
                print('Профиля {0} не найдено'.format(blade_name))
    else:
        # код для представления данных в виде массивов есди нужен
        pass
    return result