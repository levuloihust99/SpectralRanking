import random

character_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def get_unique_id(rnd: random.Random, id_len: int = 32):
    unique_id = ""
    for _ in range(id_len):
        unique_id += rnd.choice(character_pool)
    return unique_id
