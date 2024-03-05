import json
def main():
    data = [json.loads(line) for line in open('amrlib/data/annotations/dip-all-amr-daide-smosher-add.jsonl', 'r')]
    train_data = ''
    validation_data = ''
    test_data = ''
    for i in range(len(data)):
        print(i)
        if 'train' in data[i]['id']:
            if data[i]['id'].split('.')[1] != '1':
                # single = '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i-1]['sender'].capitalize()+' send to '+data[i-1]['recipient'].capitalize()+' that ' +data[i-1]['snt'] +'    '+ data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                # single = single.replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')
                # train_data += single 
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'Knowing that ' +data[i-1]['snt'] + ' ||| ' +\
                #data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            # else:
                train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
        elif 'validation' in data[i]['id']:
            # if data[i]['id'].split('.')[1] != '1':
            #     single = '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i-1]['sender'].capitalize()+' send to '+data[i-1]['recipient'].capitalize()+' that ' +data[i-1]['snt'] +'    '+ data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #     single = single.replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')
            #     validation_data += single
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i-1]['snt'] +'    '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'Knowing that ' +data[i-1]['snt'] + ' ||| ' +\
                #data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            # else:
                #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'None' + ' ||| ' +\
                #data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
        elif 'test' in data[i]['id']:
            # if data[i]['id'].split('.')[1] != '1':

            #     single = '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i-1]['sender'].capitalize()+' send to '+data[i-1]['recipient'].capitalize()+' that ' +data[i-1]['snt'] +'    '+ data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            #     single = single.replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')
            #     test_data += single
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i-1]['snt']+ '    '+'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'Knowing that ' +data[i-1]['snt'] + ' ||| ' +\
                #data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            # else:
                test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
                #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'None' + ' ||| ' +\
                #data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    with open("amrlib/data/diplomacy/data/training/train.txt", "w") as text_file:
        text_file.write(train_data)
    with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
        text_file.write(test_data)
    with open("amrlib/data/diplomacy/data/dev/validation.txt", "w") as text_file:
        text_file.write(validation_data)

if __name__ == "__main__":
    main()