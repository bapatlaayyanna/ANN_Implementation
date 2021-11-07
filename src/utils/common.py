import yaml

def read_config(config_file_path):
    with open(config_file_path) as config_file:
        content=yaml.safe_load(config_file)

        return content 