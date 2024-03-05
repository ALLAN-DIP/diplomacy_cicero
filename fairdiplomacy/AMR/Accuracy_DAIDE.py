import json
import sys
from checker import f_score
from utils import compute_accuracy

if __name__ == "__main__":
    with open('dip_daide_example.json','r') as file:
        data = json.load(file)

    count = 0
    count2 = 0 
    overall_F = 0
    overall_F2 = 0
    overall_F_noremove = 0
    overall_F2_noremove = 0
    for messages in data:
        count +=1
        msg = messages['msg']
        print(msg)
        daide = messages['gold_daide']
        daide_s = messages['translated_daide']
        daide_status = messages['daide_status']
        if daide_status == 'No-DAIDE':
            count2 += 1
        else:
            F = f_score(daide_s,daide,False)
            overall_F = overall_F + F
            print(f'better f_score is {F}')
            F2 = compute_accuracy(daide,daide_s)
            overall_F2 = overall_F2 +F2
            print(f'original f_score is {F2}')
        F_noremove = f_score(daide_s,daide,False)
        overall_F_noremove = overall_F_noremove + F_noremove
        print(f'better f_score_noremove is {F_noremove}')
        F2_noremove = compute_accuracy(daide,daide_s)
        overall_F2_noremove = overall_F2_noremove +F2_noremove
        print(f'original f_score_noremove is {F2_noremove}')
    normal_count = count-count2
    average_F = overall_F/normal_count
    average_F2 = overall_F2/normal_count
    average_F_noremove = overall_F_noremove/count
    average_F2_noremove = overall_F2_noremove/count
    print('=======================')
    print(f'all messages is {count}')
    print(f'no daide messages is {count2}')
    print(f'daide messages is {normal_count}')
    print(f'average_F is {average_F}')
    print(f'average_F2 is {average_F2}')
    print(f'average_F_noremove is {average_F_noremove}')
    print(f'average_F2_noremove is {average_F2_noremove}')