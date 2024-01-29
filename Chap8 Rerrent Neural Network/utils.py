import random
import torch
import re
import collections
from d2l import torch as d2l

d2l.DATA_HUB["time_machine"] = (
    d2l.DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def read_time_machine():
    with open(d2l.download("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]
