import numpy as np
from numpy import linalg as LA

from collections import deque

from rayuela.base.semiring import Boolean, Real
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import MinimizeState

class SCC:

    def __init__(self, fsa):
        self.fsa = fsa

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self) -> "list[frozenset]":
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """
		# Homework 3: Question 4
        scc = []
        visited = set()
        stack = []
        SCC_lst = []

        #Run DFS to visit all vertices and build a stack 
        for u in self.fsa.finish(rev=False, acyclic_check=False): # use outer loop to ganrantee every note will be visited
            if u not in visited:
                self.visit(u, visited, stack)
        
        #Reverse the WFTA
        inverG = SCC(self.fsa.reverse())
        
        #Build SCCs using a re-initialized visited and stack build from the previous DFS
        visited =  set()
        while stack:
            p = stack.pop()
            if p not in visited:
                scc = []
                inverG.buildSCC(p, visited, scc)
                SCC_lst.append(frozenset(scc))
        return SCC_lst

    def visit(self, u, visited, stack):
        visited.add(u)
        for a, outgoing, w in self.fsa.arcs(u):
            if outgoing not in visited:
                self.visit(outgoing, visited, stack)
        stack.append(u)

    def buildSCC(self, u, visited, scc):
        scc.append(u)
        visited.add(u)
        for a, outgoing, w in self.fsa.arcs(u):
            if outgoing not in visited:
                self.buildSCC(outgoing, visited, scc)
