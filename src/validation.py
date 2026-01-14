def walk_forward_split(data, train_size, test_size):
    splits = []
    start = 0

    while start + train_size + test_size <= len(data):
        train_idx = data.index[start:start + train_size]
        test_idx = data.index[start + train_size:start + train_size + test_size]

        splits.append((train_idx, test_idx))
        start += test_size

    return splits

