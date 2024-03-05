# Opening JSON file
import json
def main():
    test_data = [json.loads(line) for line in open('test.jsonl', 'r')]
    train_data = [json.loads(line) for line in open('train.jsonl', 'r')]
    val_data = [json.loads(line) for line in open('validation.jsonl', 'r')]
    data = [json.loads(line) for line in open('amrlib/data/annotations/dip-all-amr-daide-smosher.jsonl', 'r')]
    test_speakers = []
    test_receivers = []
    train_speakers = []
    train_receivers = []
    val_speakers = []
    val_receivers = []
    count = 0

    for i in range(len(data)):
        print(data[i]['id'])
        print(data[i]['snt'])
        if 'train' in data[i]['id']:
            game_id = int(data[i]['id'].split('_')[-1].split('.')[0])
            message_id = int(data[i]['id'].split('_')[-1].split('.')[1])
            message = train_data[game_id-1]['messages'][message_id-1]
            sender = train_data[game_id-1]['speakers'][message_id-1]
            recipient = train_data[game_id-1]['receivers'][message_id-1]
            season = train_data[game_id-1]['seasons'][message_id-1]
            year = train_data[game_id-1]['years'][message_id-1]
            true_game_id = train_data[game_id-1]['game_id']
            data[i]['sender'] = sender.capitalize()
            data[i]['recipient'] = recipient.capitalize()
            data[i]['season'] = season
            data[i]['year'] = year
            data[i]['game_id'] = true_game_id
        elif 'validation' in data[i]['id']:
            game_id = int(data[i]['id'].split('_')[-1].split('.')[0])
            message_id = int(data[i]['id'].split('_')[-1].split('.')[1])
            message = val_data[game_id-1]['messages'][message_id-1]
            sender = val_data[game_id-1]['speakers'][message_id-1]
            recipient = val_data[game_id-1]['receivers'][message_id-1]
            season = val_data[game_id-1]['seasons'][message_id-1]
            year = val_data[game_id-1]['years'][message_id-1]
            true_game_id = val_data[game_id-1]['game_id']
            data[i]['sender'] = sender.capitalize()
            data[i]['recipient'] = recipient.capitalize()
            data[i]['season'] = season
            data[i]['year'] = year
            data[i]['game_id'] = true_game_id
        elif 'test' in data[i]['id']:
            game_id = int(data[i]['id'].split('_')[-1].split('.')[0])
            message_id = int(data[i]['id'].split('_')[-1].split('.')[1])
            message = test_data[game_id-1]['messages'][message_id-1]
            sender = test_data[game_id-1]['speakers'][message_id-1]
            recipient = test_data[game_id-1]['receivers'][message_id-1]
            season = test_data[game_id-1]['seasons'][message_id-1]
            year = test_data[game_id-1]['years'][message_id-1]
            true_game_id = test_data[game_id-1]['game_id']
            data[i]['sender'] = sender.capitalize()
            data[i]['recipient'] = recipient.capitalize()
            data[i]['season'] = season
            data[i]['year'] = year
            data[i]['game_id'] = true_game_id

    with open('amrlib/data/annotations/dip-all-amr-daide-smosher-add-more.jsonl', "w") as outfile:
        for i in data:
            outfile.write(json.dumps(i) + "\n")




    # for i in range(len(test_data)):
    #     if i != 38:
    #         print(i)
    #         print(count)
    #         print(len(test_data[i]['speakers']))
    #         for j in range(len(test_data[i]['speakers'])):
    #             test_speakers.append(test_data[i]['speakers'][j])
    #             test_receivers.append(test_data[i]['receivers'][j])
    #             count += 1
    # for i in range(len(train_data)):
    #     if i in [1,2,3,4,7,8,9,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,39]:
    #         for j in range(len(train_data[i]['speakers'])):
    #             train_speakers.append(train_data[i]['speakers'][j])
    #             train_receivers.append(train_data[i]['receivers'][j])
    # for i in range(len(val_data)):
    #     if i !=19:
    #         for j in range(len(val_data[i]['speakers'])):
    #             val_speakers.append(val_data[i]['speakers'][j])
    #             val_receivers.append(val_data[i]['receivers'][j])
    # train_data = ''
    # validation_data = ''
    # test_data = ''
    # count_train = 0
    # count_dev = 0
    # count_test = 0
    # amr_empty = 0
    # for i in range(len(data)):
    #     if 'train' in data[i]['id']:
    #         # print('-----------')
    #         # print(data[i]['id'])
    #         # print(train_speakers[count_train].capitalize())
    #         # print(train_receivers[count_train].capitalize())
    #         # print(data[i]['snt'])
    #         # print(data[i]['amr'])
    #         # print(data[i]['amr'].replace(train_speakers[count_train].capitalize(),'SEN').replace(train_receivers[count_train].capitalize(),'REC'))
    #         train_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ train_speakers[count_train].capitalize() +' send to ' +train_receivers[count_train].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #         count_train += 1
    #         if 'amr-empty' in data[i]['amr']:
    #             amr_empty +=1
    #     elif 'validation' in data[i]['id']:
    #         validation_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ val_speakers[count_dev].capitalize() +' send to ' +val_receivers[count_dev].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #         count_dev += 1
    #     elif 'test' in data[i]['id']:
    #         test_data += '# ::id '+data[i]['id']+'\n'+'# ::snt '+ test_speakers[count_test].capitalize() +' send to ' +test_receivers[count_test].capitalize()+' that ' +data[i]['snt']+'\n'+data[i]['amr']+'\n'+'\n'
    #         count_test += 1
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
