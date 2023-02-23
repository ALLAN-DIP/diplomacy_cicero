import json
import nltk
import sacrebleu
def main():
    data_o = [json.loads(line) for line in open('annotations/dip-all-amr-daide-smosher.jsonl', 'r')]
    data_t = [json.loads(line) for line in open('annotations/test3.jsonl', 'r')]

    refs = [[]]
    sys = []
    count = 0
    count_1 = 0
    for i in range(len(data_o)):
        for j in range(len(data_t)):
            if data_o[i]['id'] == data_t[j]['id']:
                if data_o[i]['daide-status'] == 'No-DAIDE' and data_t[j]['daide-status'] == 'No-DAIDE':
                    count += 1
                    count_1 +=1

                elif data_o[i]['daide-status'] != 'No-DAIDE' and data_t[j]['daide-status'] == 'No-DAIDE':
                    count_1 += 1
                    refs[0].append(data_o[i]['daide'].replace('(','').replace(')',''))
                    sys.append('No-DAIDE')
                elif data_o[i]['daide-status'] == 'No-DAIDE' and data_t[j]['daide-status'] != 'No-DAIDE':
                    count_1 += 1
                    refs[0].append('No-DAIDE')
                    sys.append(data_t[j]['daide'].replace('(','').replace(')',''))
                else:
                    count_1 += 1
                    reference = data_o[i]['daide'].replace('(','').replace(')','')
                    hypothesis = data_t[j]['daide'].replace('(','').replace(')','')
                    refs[0].append(reference)
                    sys.append(hypothesis)
    bleu = sacrebleu.metrics.BLEU()
    print(count)
    print(count_1)
    print(bleu.corpus_score(sys, refs))
    print(bleu.get_signature())
        # if 'train' in data[i]['id']:
        #     count_train += 1
        #     train_data += '# ::id ' + data[i]['id'] + '\n' + '# ::snt ' + data[i]['snt'] + '\n' + data[i][
        #         'amr'] + '\n' + '\n'
        #     if 'amr-empty' in data[i]['amr']:
        #         amr_empty += 1
        # elif 'validation' in data[i]['id']:
        #     count_dev += 1
        #     validation_data += '# ::id ' + data[i]['id'] + '\n' + '# ::snt ' + data[i]['snt'] + '\n' + data[i][
        #         'amr'] + '\n' + '\n'
        # if 'test' in data_o[i]['id']:
        #     Daide = data_o[i]['daide-status']
            # test_data += '# ::id ' + data[i]['id'] + '\n' + '# ::snt ' + data[i]['snt'] + '\n' + data[i][
            #     'amr'] + '\n' + '\n'

    # print(count_train)
    # print(count_dev)
    # print(count_test)
    # print(amr_empty)
    #
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
