import csv
import pandas as pd


file = open('final.csv')
reader = csv.reader(file)
arr = list(reader)

file.close()

save =[]
id = 10
for row in range(len(arr)):
    if (arr[row][id] in save):
        arr[row][id] = save.index(arr[row][id])
    else:
        save.append(arr[row][id])
        arr[row][id] = save.index(arr[row][id])

with open('final.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for row in arr:
        writer.writerow(row)

# for i in arr:
#     print(i)




"""
author_type
district
street
underground
"""



