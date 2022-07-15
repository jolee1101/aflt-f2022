from frozendict import frozendict

from rayuela.base.semiring import Boolean
from rayuela.base.symbol import Sym, ε

from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State


class FST(FSA):

	def __init__(self, R=Boolean):

		# DEFINITION
		# A weighted finite-state transducer is a 8-tuple <Σ, Δ, Q, F, I, δ, λ, ρ> where
		# • Σ is an alphabet of symbols;
		# • Δ is an alphabet of symbols;
		# • Q is a finite set of states;
		# • I ⊆ Q is a set of initial states;
		# • F ⊆ Q is a set of final states;
		# • δ is a finite relation Q × Σ × Δ × Q × R;
		# • λ is an initial weight function;
		# • ρ is a final weight function.

		# NOTATION CONVENTIONS
		# • single states (elements of Q) are denoted q
		# • multiple states not in sequence are denoted, p, q, r, ...
		# • multiple states in sequence are denoted i, j, k, ...
		# • symbols (elements of Σ and Δ) are denoted lowercase a, b, c, ...
		# • single weights (elements of R) are denoted w
		# • multiple weights (elements of R) are denoted u, v, w, ...

		super().__init__(R=R)

		# alphabet of output symbols
		self.Delta = set()

	def add_arc(self, i, a, b, j, w=None):
		if w is None: w = self.R.one

		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(b, Sym): b = Sym(b)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.Delta.add(b)
		self.δ[i][(a, b)][j] += w

	def set_arc(self, i, a, b, j, w):
		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(b, Sym): b = Sym(b)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.Delta.add(b)
		self.δ[i][(a, b)][j] = w

	def freeze(self):
		self.Sigma = frozenset(self.Sigma)
		self.Delta = frozenset(self.Delta)
		self.Q = frozenset(self.Q)
		self.δ = frozendict(self.δ)
		self.λ = frozendict(self.λ)
		self.ρ = frozendict(self.ρ)

	def arcs(self, i, no_eps=False):
		for ab, T in self.δ[i].items():
			if no_eps and ab == (ε, ε):
				continue
			for j, w in T.items():
				if w == self.R.zero:
					continue
				yield ab, j, w

	def accept(self, string1, string2):
		""" determines whether a string is in the language """
		# Requires composition
		raise NotImplementedError

	def top_compose(self, fst):
    		# Homework 3: Question 3
		
		# the two machines need to be in the same semiring
		assert self.R == fst.R
		
		#trim FTSs
		_self = self.trim()
		_fst = fst.trim()

		# add initial states
		composition = FST(R=self.R)

		visited = set()
		stack = [((q11, w1), (q21, w2)) for (q11, w1), (q21, w2)  in product(_self.I, _fst.I)]
		
		for ((q1, w1), (q2, w2)) in stack:
			composition.add_I(PairState(q1, q2), _self.R.__mul__(w1, w2))
		
		self_finals = {q: w for q, w in _self.F}
		fsa_finals = {q: w for q, w in _fst.F}
	
		while stack:
			((q11,w_1), (q21, w_2)) = stack.pop()
			
			E_self=[(q11, ab1, q12, w1) for (ab1, q12, w1) in _self.arcs(q11)]
			
			E_fst=[(q21, ab2, q12, w2) for (ab2, q12, w2) in _fst.arcs(q21)]
			
			M = [((q11, ab1, q12, w1), (q21, ab2, q22, w2)) for (q11, ab1, q12, w1), (q21, ab2, q22, w2) in product(E_self, E_fst)]
			
			for ((q11, ab1, q12, w1), (q21, ab2, q22, w2)) in M:    

				ab1 = self.split_input_output(ab1)
				ab2 = self.split_input_output(ab2)
				if ab1[1]==ab2[0]:
					composition.add_arc(PairState(q11,q21), Sym(ab1[0]), Sym(ab2[1]), PairState(q12, q22), _self.R.__mul__(w1, w2))
		
				if ((q12, w_1), (q22,w_2)) not in visited:					
					stack.append(((q12, w_1), (q22, w_2)))
					visited.add(((q12, w_1), (q22, w_2)))
					
			# final state handling
			if q11 in self_finals and q21 in fsa_finals:
				composition.add_F(PairState(q11, q21), self.R.__mul__(self_finals[q11], fsa_finals[q21]))

		return composition.trim()

	def bottom_compose(self, fst):
		# Homework 3: Question 3
		return(fst.top_compose(self))	

	def split_input_output(self, ab):
		return ab.__str__()[ab.__str__().find('(')+1:ab.__str__().find(')')].split(', ')
