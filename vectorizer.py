from word_embedding import get_vec, vec_len
import torch

class Vectorizer(object):
    def __init__(self):
        pass

    def __call__(self, data):
        text, label, uniform_size = data
        label = [label.index(max(label))]
        # label = Tensor(label)
        ret = []
        # print(text)
        for word in text:
            vec = get_vec(word)
            if vec:
                ret.append(vec)

        
        if uniform_size != None:
            if len(ret) > uniform_size:
                raise Exception('proceed the data to ensure that data is shorter the uniform length')

            ret += [[0] * vec_len] * (uniform_size - len(ret))

        return torch.tensor(ret), torch.tensor(label, dtype=torch.long)
