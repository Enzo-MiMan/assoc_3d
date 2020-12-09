import os
import re
import argparse
import numpy as np
import pandas
from os.path import join, dirname
import matplotlib.pyplot as plt
import inspect


def show_trajectory(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=20)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    # traj.set_xlim(1, 5)
    # traj.set_ylim([10, 40])
    # traj.set_xticks(range(1, 5))
    # traj.set_yticks([(i*10) for i in range(1, 5)])

    plt.scatter(x, y, s=12, c='r', marker='o')
    plt.show()


