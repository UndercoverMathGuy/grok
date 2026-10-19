# Orchestrator check of VIOL_b_p11_n2_b2.5_pen_s51_9.npz (`proj.py`)

The stored network has gamma/D_{1} = 1.0000080. It is not in N_{1}, because beta_4 = 1.08e-8 != 0.

Steps:
1. Exact projection to N_{1}: zero every foreign clean coefficient by a least-norm change of the foreign output
   components. Result: gamma/D_{1} = 0.99986.
2. Local SLSQP re-optimisation inside N_{1}, with exact equality constraints beta_j = 0 for j >= 2. Result:
   gamma/D_{1} = 0.999999999. This is the IF block again.

Conclusion: not a violation of RD_{1}. The gain needs a genuinely nonzero (tiny) second feature. Measured against its
true class, N_{1,4}, the network sits at 0.24 D_{1,4}.

Side remark: a tiny clean by-product at frequency 4, produced by an interference field that is already paid for,
lifts the network above D_{1}. So W_S is not continuous under adding vanishing features. This is relevant to the
amplitude-floor form of the upper half (E_dual D10).
