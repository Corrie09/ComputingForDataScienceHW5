###########################################

#
# 1. In this exercise we will make a "Patient" class
#
# The Patient class should store the state of
# a patient in our hospital system.
#
#
# 1.1)
# Create a class called "Patient".
# The constructor should have two parameters
# (in addition to self, of course):
#
# 1. name (str)
# 2. symptoms (list of str)
#
# the parameters should be stored as attributes
# called "name" and "symptoms" respectively
class Patient:
    def __init__(self, name: str, symptoms: list):
        # store the parameters as attributes
        self.name = name
        self.symptoms = symptoms
        # initialize an attribute called "tests" as an empty dictionary to store test results
        self.tests = {}

    #
    # 1.2)
    # Create a method called "add_test"
    # which takes two paramters:
    # 1. the name of the test (str)
    # 2. the results of the test (bool)
    #
    # This information should be stored somehow.
    def add_test(self, test_name, result):
        # store the test result in the tests dictionary
        self.tests[test_name] = result

    #
    # 1.3)
    # Create a method called has_covid()
    # which takes no parameters.
    #
    # "has_covid" returns a float, between 0.0
    # and 1.0, which represents the probability
    # of the patient to have Covid-19
    #
    # The probability should work as follows:
    #
    # 1. If the user has had the test "covid"
    #    then it should return .99 if the test
    #    is True and 0.01 if the test is False
    # 2. Otherwise, probability starts at 0.05
    #    and ncreases by 0.1 for each of the
    #    following symptoms:
    #    ['fever', 'cough', 'anosmia']
    def has_covid(self):
        if "covid" in self.tests:
            # Rule 1: return .99 if test is True, .01 if False
            return 0.99 if self.tests["covid"] else 0.01
        # Rule 2: start at 0.05 and increase by 0.1 for each symptom for each symptom
        else:
            probability = 0.05  # base probability starts at 0.05
            covid_symptoms_to_check = ["fever", "cough", "anosmia"]

            for symptom in covid_symptoms_to_check:
                if symptom in self.symptoms:
                    probability += 0.1  # adding 10% for each symptom

            return probability


######################

# 2. In this exercise you will make an English Deck class made of Card classes
#
# the Card class should represent each of the cards
#
# the Deck class should represent the collection of cards and actions on them

# 2.1) Create a Card class called "Card".
# The constructor (__init__ ) should have two parameters the "suit" and the "value" and the suit of the card.
# The class should store both as attributes.


class Card:
    def __init__(self, suit: str, value: str):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"


# 2.2) Create a Deck class called "Deck".
# The constructor will create an English Deck (suits: Hearts, Diamonds, Clubs, Spades and values: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K). It will create a list of cards that contain each of the existing cards in an English Deck.
# Create a method called "shuffle" that shuffles the cards randomly.
# Create a method called "draw" that will draw a single card and print the suit and value. When a card is drawn, the card should be removed from the deck.
import random  # import random module to shuffle


class Deck:
    def __init__(self):
        # now we create the deck of cards
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        # creation of every combination of suit and value
        self.cards = []
        for suit in suits:
            for value in values:
                self.cards.append(Card(suit, value))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if len(self.cards) == 0:
            print("There are no cards left in the deck.")
            return None

        card = self.cards.pop()  # remove and return the last card
        print(f"Drew this card: {card}")
        return card


###################

# 3. In this exercise you will create an interface that will serve as template
# for different figures to compute their perimeter and surface.

# 3.1Create an abstract class (interface) called "PlaneFigure" with two abstract methods:
# compute_perimeter() that will implement the formula to compute the perimiter of the plane figure.
# compute_surface() that will implement the formula to compute the surface of the plane figure.

from abc import ABC, abstractmethod


class PlaneFigure(ABC):
    """
    Abstract base class for 2D geometric figures.
    All shapes must implement perimeter and surface calculations.
    """

    @abstractmethod
    def compute_perimeter(self):
        """Calculate the perimeter of the figure."""
        pass

    @abstractmethod
    def compute_surface(self):
        """Calculate the surface area of the figure."""
        pass


# 3.2 Create a child class called "Triangle" that inherits from "PlaneFigure" and has as parameters in the constructor "base", "c1", "c2", "h". ("base" being the base, "c1" and "c2" the other two sides of the triangle and "h" the height). Implement the abstract methods with the formula of the triangle.


class Triangle(PlaneFigure):
    def __init__(self, base: float, c1: float, c2: float, h: float):
        """
        Triangle with base, two other sides (c1, c2), and height (h).
        """
        self.base = base
        self.c1 = c1
        self.c2 = c2
        self.h = h

    def compute_perimeter(self):
        # Perimeter = sum of all sides
        return self.base + self.c1 + self.c2

    def compute_surface(self):
        # Area = (base * height)/2
        return 0.5 * self.base * self.h


# 3.3 Create a child class called "Rectangle" that inherits from "PlaneFigure" and has as parameters in the constructor "a", "b" (sides of the rectangle). Implement the abstract methods with the formula of the rectangle.


class Rectangle(PlaneFigure):
    def __init__(self, a: float, b: float):
        """
        Rectangle with sides a and b.
        """
        self.a = a
        self.b = b

    def compute_perimeter(self):
        # Perimeter = 2 * (a + b)
        return 2 * (self.a + self.b)

    def compute_surface(self):
        # Area = a * b
        return self.a * self.b


# 3.3 Create a child class called "Circle" that inherits from "PlaneFigure" and has as parameters in the constructor "radius" (radius of the circle). Implement the abstract methods with the formula of the circle.

import math


class Circle(PlaneFigure):
    def __init__(self, radius: float):
        """
        Circle with given radius.
        """
        self.radius = radius

    def compute_perimeter(self):
        # Perimeter (Circumference) = 2 * π * radius

        return 2 * math.pi * self.radius

    def compute_surface(self):
        # Area = π * radius^2

        return math.pi * (self.radius**2)
