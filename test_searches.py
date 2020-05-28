import os
from constants import (
    FILENAMES_OF_QUERIES,
    PLSH_INDEX,
    NLSH_INDEX,
    QUERIES_PATH
)
from loader import _read_dataset_names 

def build_commands():
    commands = []
    queries_list = _read_dataset_names(
        path=FILENAMES_OF_QUERIES,
        audio_path=QUERIES_PATH
    )
    for index_type in [NLSH_INDEX, PLSH_INDEX]:
        for query_name in queries_list[:100]:
            cmd = " ".join([
                f"python main.py search -i {index_type} -f {query_name}",
                "--num_permutations 1000"
            ])
            commands.append(cmd)

    return commands


def execute_commands():
    commands = build_commands()
    for c in commands:
        os.system(c)


if __name__ == "__main__":
    execute_commands()