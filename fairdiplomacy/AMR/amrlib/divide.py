# Opening JSON file
import json
def main():
    data = [json.loads(line) for line in open('amrlib/data/annotations/dip-all-amr-daide-smosher.jsonl', 'r')]
    train_data = ''
    validation_data = ''
    test_data = ''
    count_train = 0
    count_dev = 0
    count_test = 0
    amr_empty = 0
    for i in range(len(data)):
        if 'train' in data[i]['id']:
            count_train += 1
            train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
            if 'amr-empty' in data[i]['amr']:
                amr_empty +=1
        elif 'validation' in data[i]['id']:
            count_dev += 1
            validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
        elif 'test' in data[i]['id']:
            count_test += 1
            test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'

    with open("amrlib/data/diplomacy/data/training/train.txt", "w") as text_file:
        text_file.write(train_data)
    with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
        text_file.write(test_data)
    with open("amrlib/data/diplomacy/data/dev/validation.txt", "w") as text_file:
        text_file.write(validation_data)

    
    # stog = amrlib.load_stog_model()
    # graphs = stog.parse_sents(["As far as the CEO is concerned, he is the best at everything"])
    # for graph in graphs:
    #     print(graph)


if __name__ == "__main__":
    main()
