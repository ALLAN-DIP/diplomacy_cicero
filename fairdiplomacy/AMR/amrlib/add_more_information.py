import json
import glob
from collections import defaultdict
def main():
    # json_files = glob.glob('amrlib/data/moves/*.json')
    # status = dict()
    # for json_file in json_files:
    #     with open(json_file) as f:
    #         data = json.load(f)
    #         grouped_countries = {}
    #         for key, value in data['territories'].items():
    #             if value in grouped_countries:
    #                 grouped_countries[value].append(key)
    #             else:
    #                 grouped_countries[value] = [key]
    #         output_dict = {}
    #         for value, keys in grouped_countries.items():
    #             output_dict[value] = f"{value} which occupies {','.join(keys)}"

    #         combined_values = output_dict
    #     status[json_file.split('/')[-1]] = combined_values
    # print(status)
    
    
    data = [json.loads(line) for line in open('amrlib/data/annotations/dip-all-amr-daide-smosher-add-more.jsonl', 'r')]
    train_data = ''
    validation_data = ''
    test_data = ''
    for i in range(len(data)):
        if 'train' in data[i]['id']:
            if data[i]['amr'] != '(a / amr-empty)':
            # string = 'DiplomacyGame'+str(data[i]['game_id'])+'_'+data[i]['year']+'_'+data[i]['season'].lower()+'.json'
            # print(string)
            # print(status[string])
            # print(data[i]['sender'].capitalize())
            # print(data[i]['recipient'].capitalize())
            #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
            #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #if data[i]['sender'].capitalize() in status[string] and data[i]['recipient'].capitalize() in status[string]:
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ status[string][data[i]['sender'].capitalize()].replace(data[i]['sender'].capitalize(),'SEN') +' send to ' +status[string][data[i]['recipient'].capitalize()].replace(data[i]['recipient'].capitalize(),'REC')+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #else:
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'SEN' +' send to ' +'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
        elif 'validation' in data[i]['id']:
            if data[i]['amr'] != '(a / amr-empty)':
            #string = 'DiplomacyGame'+str(data[i]['game_id'])+'_'+data[i]['year']+'_'+data[i]['season'].lower()+'.json'
            #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
            #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            
            #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #if data[i]['sender'].capitalize() in status[string] and data[i]['recipient'].capitalize() in status[string]:
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+status[string][data[i]['sender'].capitalize()].replace(data[i]['sender'].capitalize(),'SEN') +' send to ' +status[string][data[i]['recipient'].capitalize()].replace(data[i]['recipient'].capitalize(),'REC')+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #else:
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+'SEN' +' send to ' +'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'SEN' +' send to ' +'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
        elif 'test' in data[i]['id']:
            if data[i]['amr'] != '(a / amr-empty)':
                test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'SEN' +' send to ' +'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #string = 'DiplomacyGame'+str(data[i]['game_id'])+'_'+data[i]['year']+'_'+data[i]['season'].lower()+'.json'
            #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
            #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['season']+' '+data[i]['year']+' | '+status[string]+' ||| '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #if data[i]['sender'].capitalize() in status[string] and data[i]['recipient'].capitalize() in status[string]:
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+status[string][data[i]['sender'].capitalize()] +' send to ' +status[string][data[i]['recipient'].capitalize()]+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+status[string][data[i]['sender'].capitalize()].replace(data[i]['sender'].capitalize(),'SEN') +' send to ' +status[string][data[i]['recipient'].capitalize()].replace(data[i]['recipient'].capitalize(),'REC')+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
            #else:
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+'In '+data[i]['year']+' '+data[i]['season']+','+'SEN' +' send to ' +'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    with open("amrlib/data/diplomacy/data/training/train.txt", "w") as text_file:
        text_file.write(train_data)
    with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
        text_file.write(test_data)
    with open("amrlib/data/diplomacy/data/dev/validation.txt", "w") as text_file:
        text_file.write(validation_data)

if __name__ == "__main__":
    main()