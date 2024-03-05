import json
def main():
    data = [json.loads(line) for line in open('amrlib/data/annotations/dip-all-amr-daide-smosher-add.jsonl', 'r')]
    with open('amrlib/data/model_parse_xfm/checkpoint-12320/dev-pred.txt','r') as text_file:
        m = text_file.read().split('\n\n')
    data_test = list()
    for i in range(len(data)):
        if 'test' in data[i]['id']:
            data_test.append({'sender':data[i]['sender'],'recipient':data[i]['recipient']})
    print(data_test)
    print(len(data_test))
    m = m[:-1]
    print(len(m))
    for i in range(len(m)):
        m[i] = m[i].replace('SEN',data_test[i]['sender']).replace('REC',data_test[i]['recipient'])
    output_test = ''
    for i in range(len(m)):
        output_test += m[i]+'\n'+'\n'
    with open("hhh.text", "w") as text_file:
        text_file.write(output_test)
    # for i in range(len(data)):
    #     if 'test' in data[i]['id']:
    #         print(data[i]['id'])
    # output = 
    # train_data = ''
    # validation_data = ''
    # test_data = ''
    # for i in range(len(data)):
    #     if 'train' in data[i]['id']:
    #         #train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
    #         train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' + data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #     elif 'validation' in data[i]['id']:
    #         #validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
    #         validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #     elif 'test' in data[i]['id']:
    #         #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ 'SEN' +' send to ' + 'REC'+' that ' +data[i]['snt'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+data[i]['amr'].replace(data[i]['sender'].capitalize(),'SEN').replace(data[i]['recipient'].capitalize(),'REC')+'\n'+'\n'
    #         test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ data[i]['sender'].capitalize() +' send to ' +data[i]['recipient'].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #         #test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    # with open("amrlib/data/diplomacy/data/training/train.txt", "w") as text_file:
    #     text_file.write(train_data)
    # with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
    #     text_file.write(test_data)
    # with open("amrlib/data/diplomacy/data/dev/validation.txt", "w") as text_file:
    #     text_file.write(validation_data)

if __name__ == "__main__":
    main()