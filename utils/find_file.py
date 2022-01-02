import os


def find_all_file(fold):
    for root, ds, fs in os.walk(fold):
        for f in fs:
            if not f.startswith('._'):
                yield f


def find_all_file_csv(fold):
    for root, ds, fs in os.walk(fold):
        for f in fs:
            if not f.startswith('._'):
                if f.endswith('.csv'):
                    yield f
