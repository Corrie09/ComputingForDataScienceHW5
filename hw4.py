##### Try to use map and reduce in the next 3 exercises
# 1)
# Create a function called "count_simba" that counts and returns
# the number of times that Simba appears in a list of
# strings. Example:
# ["Simba and Nala are lions.", "I laugh in the face of danger.",
#  "Hakuna matata", "Timon, Pumba and Simba are friends, but Simba could eat the other two."]
#

from functools import reduce


def count_simba(sentences):
    counts = map(lambda s: s.count("Simba"), sentences)
    total = reduce(lambda a, b: a + b, counts)
    return total

"""
sentences = [
    "Simba and Nala are lions.",
    "I laugh in the face of danger.",
    "Hakuna matata",
    "Timon, Pumba and Simba are friends, but Simba could eat the other two.",
]

print("Count of simba:", count_simba(sentences))  # Output: 3
"""

# 2)
# Create a function called "get_day_month_year" that takes
# a list of datetimes.date and returns a pandas dataframe
# with 3 columns (day, month, year) in which each of the rows
# is an element of the input list and has as value its
# day, month, and year.
#

import pandas as pd


def get_day_month_year(dates):
    data = list(map(lambda d: {"day": d.day, "month": d.month, "year": d.year}, dates))
    df = pd.DataFrame(data)
    return df


""" 
import datetime as dt
dates = [
    dt.date(2020, 5, 17),
    dt.date(2021, 6, 18),
    dt.date(2022, 7, 19),
]
print(get_day_month_year(dates))
"""
#################################################

# 3)
# Create a function called "compute_distance" that takes
# a list of tuple pairs with latitude and longitude coordinates and
# returns a list with the distance between the two pairs
# example input: [((41.23,23.5), (41.5, 23.4)), ((52.38, 20.1),(52.3, 17.8))]
# HINT: You can use geopy.distance in order to compute the distance
#
from geopy.distance import distance


def compute_distance(coords):
    return list(map(lambda pair: distance(pair[0], pair[1]).km, coords))

"""
example_coords = [((41.23, 23.5), (41.5, 23.4)), ((52.38, 20.1), (52.3, 17.8))]
print(compute_distance(example_coords))
"""   

#################################################
# 4)
# Consider a list that each element can be an integer or
# a list that contains integers or more lists with integers
# example: [[2], 4, 5, [1, [2], [3, 5, [7,8]], 10], 1].
# create a recursive function called "sum_general_int_list"
# that takes as input this type of list
# and returns the sum of all the integers within the lists
# for instance for list_1=[[2], 3, [[1,2],5]]
# the result should be 13
#
def sum_general_int_list(lst):
    total = 0
    for item in lst:
        if isinstance(item, int):
            total += item
        elif isinstance(item, list):
            total += sum_general_int_list(item)
    return total

"""
example_list = [[2], 4, 5, [1, [2], [3, 5, [7, 8]]], 1]
print("Sum of general int list:", sum_general_int_list(example_list)) 
example_list_2 = [[2], 3, [[1, 2], 5]]
print("Sum of general int list 1:", sum_general_int_list(example_list_2)) 
"""