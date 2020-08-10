from gpiozero import CPUTemperature
import csv
from datetime import datetime as dt
import time

cpu = CPUTemperature()

try:
    with open("temp.csv",'w') as csvfile:
        writer = csv.writer(csvfile)
        while(True):
            temp = cpu.temperature
            cur_time = dt.now().strftime('%H:%M:%S')
            writer.writerow([cur_time, temp])
            print(cur_time, '{:.2f} C'.format(temp))
            time.sleep(5)
except KeyboardInterrupt:
    pass
