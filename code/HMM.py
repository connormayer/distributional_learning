from collections import defaultdict
from numpy.random import choice

# Start and end states
START = None
END = None

"""
A set of classes that allows the definition of a Hidden Markov Model 
to probabilistically generate a language corpus.
"""

class State():
    """
    A state in the HMM.
    """
    def __init__(self, state_num, label=None):
        """
        state_num: The unique numeric identifier for the state.
        label: The optional, human-readable label for the state.
        """
        self.num = state_num
        self.label = label

class Transition():
    """
    A transition in the HMM.
    """
    def __init__(self, start_state, end_state, emissions, probability):
        """
        start_state: A State object that is the starting state for this 
                     transition.
        end_state: A State object that is the state reached after taking the
                   transition.
        emissions: A list of Emission objects specifying the character(s)
                   emitted when taking this transition, and their
                   probabilities.
        probability: The probability of taking this transition.
        """
        self.start_state = start_state
        self.end_state = end_state
        self.emissions = emissions
        self.probability = probability

class Emission():
    """
    A class representing a string emitted when taking a transition.
    """
    def __init__(self, string, probability):
        """
        string: The string to emit.
        probability: The probability of emitting this string.
        """
        self.string = string
        self.probability = probability

class HMM():
    """
    A class implementing a simple Hidden Markov Model.
    """

    def __init__(self):
        self._states = {}
        self._transitions = defaultdict(list)

    def add_state(self, state_num, label=None):
        """
        Adds a state to the HMM.

        state_num: A unique integer identifying the state.
        label: An optional, human-readable label for the state.
        """
        if self._states.get(state_num):
            raise "Duplicate state number {}".format(state_num)
        new_state = State(state_num, label)
        self._states[state_num] = new_state

    def add_transition(self, start_state, end_state, emissions, probability):
        """
        Adds a transition to the HMM.

        start_state: A State object that is the starting state for this 
                     transition.
        end_state: A State object that is the state reached after taking the
                   transition.
        emissions: A list of Emission objects specifying the character(s)
                   emitted when taking this transition, and their
                   probabilities.
        probability: The probability of taking this transition.
        """
        emissions = [Emission(string, prob) for string, prob in emissions]
        new_transition = Transition(
            start_state, end_state, emissions, probability
        )
        self._transitions[start_state].append(new_transition)

    def generate_output(self):
        """
        Returns the output of generating a single string using the HMM.
        """
        start_transitions = self._transitions[START]
        output = []
        transition = self.get_next_element(start_transitions)
        while transition.end_state is not END:
            emission = self.get_next_element(transition.emissions).string
            if emission:
                output.append(emission)
            next_transitions = self._transitions[transition.end_state]
            transition = self.get_next_element(next_transitions)
        return output

    def generate_stringset(self, n):
        """
        Generates a list of strings using the HMM.

        n: The number of strings to generate.
        """
        stringset = []
        for _ in range(n):
            stringset.append(self.generate_output())
        return stringset

    def get_next_element(self, elements):
        """
        Samples an element from the provided list based on their associated
        probabilities.

        elements: A list of elements to sample from. Elements can be any type,
                  but must have a 'probability' attribute.
        """
        element_probabilities = [e.probability for e in elements]
        total_probability = sum(element_probabilities)
        if total_probability != 1:
            element_probabilities = list(map(
                lambda x: x / total_probability, element_probabilities
            ))
        next_element = choice(elements, p=element_probabilities)
        return next_element
