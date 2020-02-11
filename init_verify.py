# if __name__ == "__main__":
#     pass
#     import random
#     with open('data/data.train', 'r') as f:
#         lines = f.readlines()

#     random.shuffle(lines)

#     verify_size = len(lines) // 10

#     for name, data in { 'verify': lines[:verify_size], 'train': lines[verify_size:] }.items():
#         with open('data/data.%s' % name, 'w') as f:
#             f.writelines(data)
    