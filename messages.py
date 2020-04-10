from constants import (
    CREATE_INDEX,
    METHODS
)


def invalid_method_msg(method_name):
    message = "Method '{}' is invalid. Valid methods are: {}".format(
        method_name,
        METHODS
    )
    return message


def has_no_dumped_files_msg():
    message = ''.join([
        "ERROR: Couldn't load inverted index or audio mapping. ",
        "Are they dumped as files at all? ",
        "Use method '{}' ".format(CREATE_INDEX),
        "to generate the inverted index and audio mapping first."
    ])
    return message
