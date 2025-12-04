from cs336_basics.pretokenization import pre_tokenize
from typing import Optional


class PreToken:
    def __init__(self, word: bytes, count: int):
        self.count = count
        self.word = word
        self.tokens = [word[i:i + 1] for i in range(len(word))]
        # print(self.tokens)


class VocabList:
    def __init__(self, vocab: list[bytes]):
        self.vocab: list[bytes] = vocab
        self.vocab_set = set(vocab)

    def add(self, word: bytes) -> None:
        self.vocab_set.add(word)
        self.vocab.append(word)

    def get_vocab_dict(self) -> dict[int, bytes]:
        vocab_dict: dict[int, bytes] = {i: s for i, s in enumerate(self.vocab)}
        return vocab_dict

    def exists(self, word: bytes) -> bool:
        return word in self.vocab_set


def find_largest_token(count_dict: dict[tuple[bytes, bytes], int]) -> Optional[tuple[bytes, bytes]]:
    if len(count_dict) == 0:
        return None
    #max_token, max_count = max(
    #    count_dict.items(),
    #    key=lambda kv: (
    #        kv[1],  # highest count first
    #        (kv[0][0] + kv[0][1])  # smallest lexicographic combined bytes
    #    )
    #)
    max_count = float('-inf')
    max_token = None
    for token, count in count_dict.items():
        if count > max_count:
            max_token = token
            max_count = count
        elif count == max_count:
            if token > max_token:
                max_token = token

    if max_count <= 0:
        return None
    return max_token


def decrease_cnt(token: tuple[bytes, bytes], val: int, count_dict: dict[tuple[bytes, bytes], int]):
    if token in count_dict:
        # print(f"token : {token},v = {val} before : {count_dict[token]}")
        count_dict[token] -= val
        # print(f"token : {token} after : {count_dict[token]}")
        if count_dict[token] <= 0:
            count_dict.pop(token)
        # print(count_dict)


def increase_cnt(token: tuple[bytes, bytes], pre_token: PreToken, count_dict: dict[tuple[bytes, bytes], int],
                 token_idx: dict[bytes, set[PreToken]]):
    if token not in count_dict:
        count_dict[token] = 0
    count_dict[token] += pre_token.count
    token_str = token[0] + token[1]
    if token_str not in token_idx:
        token_idx[token_str] = set()
    token_idx[token_str].add(pre_token)


def replace_iter(largest_token: tuple[bytes, bytes], pre_token: PreToken, count_dict: dict[tuple[bytes, bytes], int],
                 token_idx: dict[bytes, set[PreToken]]):
    found = False
    ff = False
    if largest_token[0] == b"n" and largest_token[1] == b"d":
        ff = True

    for i in range(len(pre_token.tokens) - 1):
        if largest_token[0] + largest_token[1] != pre_token.tokens[i] + pre_token.tokens[i + 1]:
            continue
        if i > 0:
            # print(f"new token = {new_token}, pre = {pre_token.tokens[i - 1]}, largest = {largest_token}")
            #if pre_token.tokens[i - 1] == b"a" and pre_token.tokens[i] == b"nd":
            #    ff = True
            #    print(f"DDDD!!!! largest = {largest_token}, v = {pre_token.count}, word = {pre_token.word}")
            #    print(f"XXXX!!!! before = {count_dict[(pre_token.tokens[i - 1], pre_token.tokens[i])]}")
            decrease_cnt((pre_token.tokens[i - 1], pre_token.tokens[i]), pre_token.count, count_dict)
            increase_cnt((pre_token.tokens[i - 1], largest_token[0] + largest_token[1]), pre_token, count_dict,
                         token_idx)
            if ff:
                print(f"XXXX!!!! increase = {(pre_token.tokens[i - 1], largest_token[0] + largest_token[1])}, v = {pre_token.count}")
        if i < len(pre_token.tokens) - 2:
            #ff = False
            #if pre_token.tokens[i + 1] == b"a" and pre_token.tokens[i+2] == b"nd":
            #    print(f"RRRR!!!! largest = {largest_token}, v = {pre_token.count}, word = {pre_token.word}")
            #    print(f"XXXX!!!! before = {count_dict[(pre_token.tokens[i + 1], pre_token.tokens[i + 2])]}")
            #    ff = True
            decrease_cnt((pre_token.tokens[i + 1], pre_token.tokens[i + 2]), pre_token.count, count_dict)
            #if ff:
            #    print(f"XXXX!!!! after = {count_dict[(pre_token.tokens[i + 1], pre_token.tokens[i + 2])]}")
            increase_cnt((largest_token[0] + largest_token[1], pre_token.tokens[i + 2]), pre_token, count_dict,
                         token_idx)
            if ff:
                print(f"XXXX!!!! increase = {(largest_token[0] + largest_token[1], pre_token.tokens[i + 2])}, v = {pre_token.count}")
        # print(f"decreasing token {largest_token} by {pre_token.count}")
        decrease_cnt(largest_token, pre_token.count, count_dict)
        pre_token.tokens = pre_token.tokens[:i] + [pre_token.tokens[i] + pre_token.tokens[i + 1]] + pre_token.tokens[
            i + 2:]
        found = True
        break
    return found


def replace(largest_token: tuple[bytes, bytes], pre_token: PreToken, count_dict: dict[tuple[bytes, bytes], int],
            token_idx: dict[bytes, set[PreToken]]):
    # print(f"replacing token : {pre_token.tokens}, largest_token = {largest_token}")
    ff = False
    while True:
        res = replace_iter(largest_token, pre_token, count_dict, token_idx)
        if not res:
            break


def merge(count_dict: dict[tuple[bytes, bytes], int], vocab: VocabList, token_idx: dict[bytes, set[PreToken]],
          merge_list: list[tuple[bytes, bytes]]) -> bool:
    largest_token_tuple = find_largest_token(count_dict)
    if largest_token_tuple is None:
        return False
    #print(f"Merging vocab : {largest_token_tuple[0]} + {largest_token_tuple[1]}")
    largest_token = largest_token_tuple[0] + largest_token_tuple[1]
    if vocab.exists(largest_token):
        if largest_token_tuple in count_dict:
            count_dict.pop(largest_token_tuple)
        return True
    #print(f"Adding vocab : {largest_token} , type : {type(largest_token)}")
    vocab.add(largest_token)
    merge_list.append(largest_token_tuple)

    if largest_token not in token_idx:
        print(f"Error !!!!, largest_token : {largest_token} not found in token_idx")
        return True

    for pre_token in token_idx[largest_token]:
        replace(largest_token_tuple, pre_token, count_dict, token_idx)

    token_idx.pop(largest_token)
    if largest_token in count_dict:
        count_dict.pop(largest_token_tuple)
    return True


INITIAL_VOCAB_SIZE: int = 256


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab: list[bytes] = [bytes([i]) for i in range(INITIAL_VOCAB_SIZE)]

    vocab.extend([s.encode("utf-8") for s in special_tokens])

    vocab_list = VocabList(vocab)

    # print(f"vocab: {vocab}")

    pre_tokens = pre_tokenize(input_path, 4, special_tokens)
    token_idx: dict[bytes, set[PreToken]] = {}
    count_dict: dict[tuple[bytes, bytes], int] = {}

    idx = 0
    for k, v in pre_tokens.items():
        idx += 1
        ff = False
        if k == " Wherefore":
            print(f"parsing token {idx}th%: {k}: {v}")
            ff = True
        tokens = []
        k_bytes = k.encode("utf-8")
        if len(k_bytes) < 2:
            if not vocab_list.exists(k_bytes):
                print(f"!!!! Error one byte token {k_bytes} is not in vocab")
            continue

        for i in range(len(k_bytes) - 1):
            tokens.append((k_bytes[i:i + 1], k_bytes[i + 1:i + 2]))
            if ff:
                print(f"HHHH parsing {(k_bytes[i:i + 1], k_bytes[i + 1:i + 2])}, v = {v}")
        pre_token = PreToken(k_bytes, v)
        for token_tuple in tokens:
            token = token_tuple[0] + token_tuple[1]
            if token not in token_idx:
                token_idx[token] = set()
            if ff and token == b"re":
                print(f"HHHH before c = {len(token_idx[token])}")
            token_idx[token].add(pre_token)
            if ff and token == b"re":
                print(f"HHHH after c = {len(token_idx[token])}")

            if token_tuple not in count_dict:
                count_dict[token_tuple] = 0
            count_dict[token_tuple] += v

    # print(count_dict)
    # for k, v in token_idx.items():
    #    print(f"{k} : {len(v)}")

    merge_count = vocab_size - INITIAL_VOCAB_SIZE - len(special_tokens)
    merge_count = max(merge_count, 0)
    print(f"LOG: start merging {merge_count} iterations")
    for i in range(merge_count):
        res = merge(count_dict, vocab_list, token_idx, merges)
        if not res:
            print(f"Warning, merge completed before reaching the target number of vocab, current idx = {i}")
        tt = (b" ", b"d")
        if tt in count_dict:
            print(f"#### idx = {i},  d = {count_dict[tt]}")
        tt2 = (b" a", b"nd")
        if tt2 in count_dict:
            print(f"#### idx = {i},  a nd = {count_dict[tt2]}")

    vocab_dict = vocab_list.get_vocab_dict()
    print(vocab_dict)
    print(merges)
    return vocab_dict, merges


if __name__ == "__main__":
    #train_bpe("./tests/fixtures/tinystories_sample.txt", 400, ["<|endoftext|>"])
    #train_bpe("./tests/fixtures/simple_test.txt", 400, ["<|endoftext|>"])
    train_bpe("./tests/fixtures/corpus.en", 272, ["<|endoftext|>"])

    count_dict: dict[tuple[bytes, bytes], int]  = {}
    #k1 = (b" a", b"nd")
    #k2 = (b" ", b"d")
    #k1 = (b"s", b"t")
    #k2 = (b"e", b"d")
    #count_dict[k1] = 100
    #count_dict[k2] = 100
    #res = find_largest_token(count_dict)
    #print(res)