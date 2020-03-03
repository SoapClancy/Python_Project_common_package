from enum import Enum, unique


@unique
class SeasonTemplate1(Enum):
    spring = (3, 4, 5)
    summer = (6, 7, 8)
    autumn = (9, 10, 11)
    winter = (12, 1, 2)
