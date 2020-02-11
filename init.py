# word_dict = {}
# vec_dict = {}

# from sys import stderr

# def get_id(word):
#     global word_dict
#     word_dict
#     if word in word_dict:
#         return word_dict[word]

#     ret = len(word_dict)
#     word_dict[word] = len(word_dict)
#     return ret

# def dump_dict():
#     global word_dict
#     with open('data/dict.txt', 'w', encoding='utf-8') as f:
#         for word, word_id in word_dict.items():
#             try:
#                 f.write('%d %s\n' % (word_id, ' '.join(vec_dict[word])))
#             except KeyError:
#                 f.write('%d 0\n' % word_id)

# if __name__ == "__main__":
#     pass
#     for suf in ['.test', '.train']:
#         with open('data/sina/sinanews%s' % suf, 'r', encoding='utf-8') as fin:
#             with open('data/data%s' % suf, 'w', encoding='utf-8') as fout:
#                 for line in fin.readlines():
#                     time, label, text = line.strip().split('\t')
#                     label = [int(x.split(':')[1]) for x in label.split(' ')[1:]]
#                     # print(label)
#                     text = text.split(' ')
#                     text = list(map(get_id, text))
#                     fout.write('%s\t%s\n' % (' '.join(map(str, text)), ' '.join(map(str, label))))

#                     # break
#     stderr.write('parse over\n')

#     # with open('data/sgns.sogou.word', 'r', encoding='utf-8') as f:
#     #     n, length = map(int, f.readline().strip().split(' '))
#     #     for i in range(n):
#     #         line = f.readline().strip().split(' ')
#     #         word, vec = line[0], line[1:]
#     #         if word in word_dict:
#     #             vec_dict[word] = vec
#     #             # break

#     #         stderr.write('\r%f %%       ' % (i * 100/ n))

#     # dump_dict()