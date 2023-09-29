import spot
import numpy as np

class Specification:

    def __init__(self, formula=None, hoa=None, reward=1):

        # Spot Formula
        self.reward = reward

        if formula:
            self.formula = spot.from_ltlf(formula)
            # Translate Automaton
            self.automaton = self.formula.translate('small', 'buchi', 'sbacc')
            self.automaton = spot.to_finite(self.automaton)     
        elif hoa:
            for a in spot.automata(hoa):
                self.automaton = a
        else:
            raise("Required LTL formula or HOA.")

        # Building automaton
        self.init = self.automaton.get_init_state_number()
        self.state = self.init
        self.num_states = self.automaton.num_states()

        self.bdict = self.automaton.get_dict()

        self.variables = self.get_variables()
        self.transitions = {}
        self.acceptances = {}

        for state in range(0, self.num_states):

            self.transitions[state] = []

            self.acceptances[state] = True if self.automaton.state_acc_sets(state) else False

            for transition in self.automaton.out(state):

                self.transitions[state].append({
                    "dst": transition.dst,
                    "cond": self.map_bdd(transition.cond),
                    "cond_str": spot.bdd_format_formula(self.bdict, transition.cond)
                })

    def reset(self):
        self.state = self.init
        encoded_state = self.encode_state(self.state)
        return encoded_state

    def __len__(self):
        return self.num_states

    def __str__(self):
        str_ = ""
        for s in range(0, self.num_states):
            str_ += "\n State {}:".format(s)
            str_ += "\n  acc sets = {}".format(self.acceptances[s])
            for t in self.transitions[s]:
                str_ += "\n  edge({} -> {})".format(s, t["dst"])
                str_ += "\n    label = {}".format(t["cond_str"])
                
        return str_

    def encode_state(self, state):
        encoded_state = np.zeros((self.__len__()))
        encoded_state[state] = 1
        return encoded_state

    
                
    def map_bdd(self, cond):
        cond = spot.bdd_format_formula(self.bdict, cond)
    
        cond = cond.replace("!", " not ")
        cond = cond.replace("|", " or ")
        cond = cond.replace("&", " and ")        

        variables = ", ".join(self.variables)
        cond = eval("lambda {}: {}".format(variables, cond))
        
        return cond

    def get_variables(self):
        variables = [str(a) for a in self.automaton.ap()]
        variables += ["alive"]
        return set(variables)

    def step(self, *arglist, **keywords):
        
        for transition in self.transitions[self.state]:
            if transition["cond"](**keywords):
                self.state = transition["dst"]  

        reward = 0
        if self.acceptances[self.state]: reward=self.reward

        encoded_state = self.encode_state(self.state)
            
        return encoded_state, reward, False