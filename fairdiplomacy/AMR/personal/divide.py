import json
def main():
    test_data = [json.loads(line) for line in open('test.jsonl', 'r')]
    train_data = [json.loads(line) for line in open('train.jsonl', 'r')]
    val_data = [json.loads(line) for line in open('validation.jsonl', 'r')]
    test_speakers = []
    test_receivers = []
    train_speakers = []
    train_receivers = []
    val_speakers = []
    val_receivers = []
    for i in range(len(test_data)):
        for j in range(len(test_data[i]['speakers'])):
            test_speakers.append(test_data[i]['speakers'][j])
            test_receivers.append(test_data[i]['receivers'][j])
    for i in range(len(train_data)):
        for j in range(len(train_data[i]['speakers'])):
            train_speakers.append(train_data[i]['speakers'][j])
            train_receivers.append(train_data[i]['receivers'][j])
    for i in range(len(val_data)):
        for j in range(len(val_data[i]['speakers'])):
            val_speakers.append(val_data[i]['speakers'][j])
            val_receivers.append(val_data[i]['receivers'][j])
    #     if 'train' in data[i]['id']:
    #         count_train += 1
    #         train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #         if 'amr-empty' in data[i]['amr']:
    #             amr_empty +=1
    #     elif 'validation' in data[i]['id']:
    #         count_dev += 1
    #         validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #     elif 'test' in data[i]['id']:
    #         count_test += 1
    #         test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'

    # with open("amrlib/data/diplomacy/data/training/train.txt", "w") as text_file:
    #     text_file.write(train_data)
    # with open("amrlib/data/diplomacy/data/test/test.txt", "w") as text_file:
    #     text_file.write(test_data)
    # with open("amrlib/data/diplomacy/data/dev/validation.txt", "w") as text_file:
    #     text_file.write(validation_data)

    
    # stog = amrlib.load_stog_model()
    # graphs = stog.parse_sents(["As far as the CEO is concerned, he is the best at everything"])
    # for graph in graphs:
    #     print(graph)


if __name__ == "__main__":
    main()