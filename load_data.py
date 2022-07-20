import os


def load_data(path):

    files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]

    return files
