import random

def sample_type(type=None):
    # type = [height, weight, open, close, toggle_on, toggle_off]
    # 0.5 offset
    random_num = random.random()
    # full capability
    height = 1
    weight = 1
    open = 1
    close = 1
    toggle_on = 1
    toggle_off = 1

    if type is not None:
        if type == 0:
            height = 0.2
        elif type == 1:
            weight = 0.1
        elif type == 2:
            open = 0
        elif type == 3:
            close = 0
        elif type == 4:
            toggle_on = 0
        elif type == 5:
            toggle_off = 0
        return [height, weight, open, close, toggle_on, toggle_off]

    # 
    # weight = 0.1

    # height = random.uniform(0.2, 0.8)

    # for data sampler: 
    # if random_num < 0.333:
    #     pass
    # elif random_num < 0.444:
    #     height = random.uniform(0.2, 0.8)
    # elif random_num < 0.555:
    #     weight = random.uniform(0.1, 0.7)
    # elif random_num < 0.666:
    #     open = random.uniform(0, 0.49)
    # elif random_num < 0.777:
    #     close = random.uniform(0, 0.49)
    # elif random_num < 0.888:
    #     toggle_on = random.uniform(0, 0.49)
    # else:
    #     toggle_off = random.uniform(0, 0.49)


    if random_num < 0.143:
        pass
    elif random_num < 0.286:
        height = random.uniform(0.2, 0.8)
    elif random_num < 0.429:
        weight = random.uniform(0.1, 0.7)
    elif random_num < 0.571:
        open = random.uniform(0, 0.49)
    elif random_num < 0.714:
        close = random.uniform(0, 0.49)
    elif random_num < 0.857:
        toggle_on = random.uniform(0, 0.49)
    else:
        toggle_off = random.uniform(0, 0.49)

    # pure random
    # height = random.uniform(0.2, 0.8)
    # weight = random.uniform(0.1, 0.7)
    # open = random.random()
    # close = random.random()
    # toggle_on = random.random()
    # toggle_off = random.random()
    return [height, weight, open, close, toggle_on, toggle_off]
