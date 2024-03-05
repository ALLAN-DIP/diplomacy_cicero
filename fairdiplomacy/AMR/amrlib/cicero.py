# Opening JSON file
import json
# ::id dip_test_0001.1
def main():
    data = [json.loads(line) for line in open('RUSSIA_0.json', 'r')]
    test_data = ''
    count = 0
    for i in data[0]['phases']:
        for j in i['messages']:
            if '\n' in j['message']:
                temp = j['message'].split('\n')
                temp.remove('')
                temp = " ".join(temp)
                j['message'] = temp
            count +=1
            #test_data += '# ::id ' + 'dip_test_0001'+'.'+ str(count) + '\n' + '# ::snt ' +  j['message'] + '\n' + '(a / amr-empty)' + '\n' + '\n'
            test_data += '# ::id ' + 'dip_test_0001' + '.' + str(count) + '\n' + '# ::snt ' + j['message'] + '\n' + '(a / amr-empty)' + '\n' + '\n'
    with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
        text_file.write(test_data)

if __name__ == "__main__":
    main()