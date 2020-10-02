import numpy as np
import pandas as pd
import os
import xlwt
import time
from bjontegaard_metric import BD_RATE

# os.system('pause')

# get time stamp
time_stamp = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime(time.time()))

items = ['Seq name', 'target_rate', 'actual_rate', 'Y-PSNR', 'U-PSNR', 'V-PSNR', 'encoding_time']
added_items = []

config_file_name = 'log_config.txt'
with open(config_file_name, 'r') as fp:
    
    while True:
        line = fp.readline()
        line = line.strip()

        if not line:
            continue
        
        if line.startswith('logs path:'):
            before, logs_path = line.split('logs path:')
            logs_path = logs_path.strip()
            #print(logs_path)
        elif line.startswith('reflogs path:'):
            before, reflogs_path = line.split('reflogs path:')
            reflogs_path = reflogs_path.strip()
            #print(logs_path)
        elif line.startswith('output path:'):
            before, output_path = line.split('output path:')
            output_path = output_path.strip()
            #print(output_path)
        elif line.startswith('***end***'):
            break
fp.close()


def readlog(logs_path):

    logs = os.listdir(logs_path)
    dic = {}
    row = 1
    for log_name in logs:
        log_str = os.path.splitext(log_name)
        if log_str[1] != '.log':
            continue

        file = open(logs_path + log_name, 'r')
        tmp = log_str[0].split('_')
        seq_name = '_'.join(tmp[:-1])
        qp = tmp[len(tmp)-1]


        line = ' '
        while not line.startswith('Total Time:'):
            line = file.readline()
            line = line.strip()
            if not line.startswith('LayerId'):
                continue
            else:
                line = file.readline()
                line = file.readline()
                line = line.strip()
                nums = line.split()
                #print(nums)
                actual_rate = nums[2]
                Y_PSNR = nums[3]
                

                while True:
                    line = file.readline()
                    line = line.strip()
                    if line.startswith('Total Time:'):
                        before, after = line.split('Total Time:')
                        time, after = after.split('sec. [user]')
                        time = time.strip()
                        dic[seq_name + qp] = [actual_rate, Y_PSNR, time]
                        break

        row = row + 1 # next file
    return dic

test_dic = readlog(logs_path)
ref_dic = readlog(reflogs_path)

n = 0
r1 = np.zeros(4)
psnr1 = np.zeros(4)
r2 = np.zeros(4)
psnr2 = np.zeros(4)
time_ratio = np.zeros(4)

dic = {}
avr_bdbr = 0
avr_time_save = 0
for seq, data in ref_dic.items():
    
    r1[n%4] = float(data[0])
    r2[n%4] = float(test_dic[seq][0])

    psnr1[n%4] = float(data[1])
    psnr2[n%4] = float(test_dic[seq][1])

    ref_time  = float(data[2])
    time = float(test_dic[seq][2])
    time_ratio[n%4] = (ref_time - time) / ref_time

    if n%4 == 3:
        bdrate = BD_RATE(r1, psnr1, r2, psnr2)
        time_save = time_ratio.sum() / 4
        dic[seq] = [bdrate, time_save]
        avr_bdbr = avr_bdbr +  bdrate
        avr_time_save = avr_time_save +  time_save

    n = n + 1


csv = open(output_path + time_stamp + 'psnr.csv', 'w')
csv.write("Seq,BDBR,TS" + '\n')

csv.write('\n')
csv.write("Class A1,," + '\n')
for seq, data in dic.items():
    if 'Tango2' in seq:
        csv.write('Tango2' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'FoodMarket4' in seq:
        csv.write('FoodMarket4' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'Campfire' in seq:
        csv.write('Campfire' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

csv.write('\n')
csv.write("Class A2,," + '\n')
for seq, data in dic.items():
    if 'CatRobot' in seq:
        csv.write('CatRobot' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'DaylightRoad2' in seq:
        csv.write('DaylightRoad2' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'ParkRunning3' in seq:
        csv.write('ParkRunning3' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

csv.write('\n')
csv.write("Class B,," + '\n')
for seq, data in dic.items():
    if 'MarketPlace' in seq:
        csv.write('MarketPlace' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'RitualDance' in seq:
        csv.write('RitualDance' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'Cactus' in seq:
        csv.write('Cactus' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'BasketballDrive' in seq:
        csv.write('BasketballDrive' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'BQTerrace' in seq:
        csv.write('BQTerrace' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

csv.write('\n')
csv.write("Class C,," + '\n')
for seq, data in dic.items():
    if 'BasketballDrill' in seq:
        csv.write('BasketballDrill' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'BQMall' in seq:
        csv.write('BQMall' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'PartyScene' in seq:
        csv.write('PartyScene' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'RaceHorsesC' in seq:
        csv.write('RaceHorsesC' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

csv.write('\n')
csv.write("Class D,," + '\n')
for seq, data in dic.items():
    if 'BasketballPass' in seq:
        csv.write('BasketballPass' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'BQSquare' in seq:
        csv.write('BQSquare' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'BlowingBubbles' in seq:
        csv.write('BlowingBubbles' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'RaceHorses' in seq:
        csv.write('RaceHorses' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

csv.write('\n')
csv.write("Class E,," + '\n')
for seq, data in dic.items():
    if 'FourPeople' in seq:
        csv.write('FourPeople' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'Johnny' in seq:
        csv.write('Johnny' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')
for seq, data in dic.items():
    if 'KristenAndSara' in seq:
        csv.write('KristenAndSara' + ',' +  str(data[0]) + ',' +  str(data[1]) + '\n')

avr_bdbr = avr_bdbr /  23       
avr_time_save = avr_time_save /  23  
csv.write('\n')
csv.write('avr' + ',' +  str(avr_bdbr) + ',' +  str(avr_time_save) + '\n')
csv.close()
#df = pd.DataFrame(dic)
#df.T.to_csv(output_path + 'psnr.csv')






