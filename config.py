import configparser

config = configparser.ConfigParser()
config.read('config/config.ini')

def get_resolution_list():
    return config['Preprocessing']['resolution_list']

def get_band_list():
    return config['Preprocessing']['band_list']

def get_collection_name():
    return config['Preprocessing']['collection_name']

if __name__ == "__main__":
    print(get_resolution_list())
    print(get_band_list())
    print(get_collection_name())