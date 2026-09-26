import Heracles
-- `#print axioms` for every main theorem.  Run on the pod: `lake env lean scripts/Axioms.lean`
-- Defs / programme
#print axioms Heracles.exists_maximizer
#print axioms Heracles.isMaximizer_iff
#print axioms Heracles.GammaP_pos
-- prop:ablation
#print axioms Heracles.prop_ablation
#print axioms Heracles.ablation
#print axioms Heracles.ablation_fraction
#print axioms Heracles.share_bounds
-- prop:bottleneck
#print axioms Heracles.prop_bottleneck
#print axioms Heracles.prop_bottleneck_network
#print axioms Heracles.bottleneck_abstract
#print axioms Heracles.blockTable_gap
-- prop:shapes
#print axioms Heracles.prop_shapes_vertex
#print axioms Heracles.maximizer_eq_vertex
#print axioms Heracles.isMaximizer_iff_minVertex
#print axioms Heracles.mem_extremePoints_of_isMin
#print axioms Heracles.minimizer_isVertex
#print axioms Heracles.exists_minimizer
#print axioms Heracles.vertex_eq_zero_of_active
#print axioms Heracles.finite_vertices
#print axioms Heracles.expsum_card_zeros_le
#print axioms Heracles.expsum_zeros_finite
#print axioms Heracles.switches_finite
#print axioms Heracles.ties_of_multiset_eq
#print axioms Heracles.prop_shapes_switches
#print axioms Heracles.maximizers_between_switches
#print axioms Heracles.prop_shapes_shares
-- thm:scaling (programme level) and its lemmas
#print axioms Heracles.one_le_I0
#print axioms Heracles.I0_le_exp
#print axioms Heracles.I0_ge_of_one_le
#print axioms Heracles.psi_nonneg
#print axioms Heracles.psi_le
#print axioms Heracles.posfourier
#print axioms Heracles.psisum_K
#print axioms Heracles.psisum
#print axioms Heracles.twoterms
#print axioms Heracles.worst_le_of_psi
#print axioms Heracles.GammaP_upper
#print axioms Heracles.Kbound
#print axioms Heracles.exp_sub_one_ge_sq
#print axioms Heracles.card_supp_ge
#print axioms Heracles.share_ge
#print axioms Heracles.share_ge_gamma
#print axioms Heracles.card_supp_le
#print axioms Heracles.card_supp_le_ceil
#print axioms Heracles.dense_only_if
#print axioms Heracles.sum_cos_freq
#print axioms Heracles.exists_good_draw
#print axioms Heracles.exists_alloc_lower
#print axioms Heracles.GammaP_lower
#print axioms Heracles.GammaP_lower9
-- thm:scaling (network level) and cor:few
#print axioms Heracles.thm_scaling_net
#print axioms Heracles.thm_scaling_network
#print axioms Heracles.gammaStar_le
#print axioms Heracles.gammaStar_ge
#print axioms Heracles.cor_few
#print axioms Heracles.few_bound
#print axioms Heracles.tendsto_gK
#print axioms Heracles.cor_few_prod
#print axioms Heracles.cor_few_ratio_prod
-- Section approach
#print axioms Heracles.lem_avg
#print axioms Heracles.spectrum
#print axioms Heracles.spectrum_avg
#print axioms Heracles.lem_phase
#print axioms Heracles.prop_coverage
#print axioms Heracles.beta_netTable
#print axioms Heracles.thm_universal
#print axioms Heracles.thm_universal_unit
#print axioms Heracles.thm_universal_Cb
#print axioms Heracles.budget_le_Cb
#print axioms Heracles.thm_universal_prod
-- cor:dense
#print axioms Heracles.sum_cov_offsets
#print axioms Heracles.GammaP_one
#print axioms Heracles.dense_maximizer_iff
#print axioms Heracles.dense_constant
#print axioms Heracles.dense_network_equal_amp
-- lem:cost and sub-lemmas
#print axioms Heracles.Cost.chain_lemma
#print axioms Heracles.Cost.det_one_add_smul_expand
#print axioms Heracles.Cost.lem_hadamard
#print axioms Heracles.Cost.prop_esym
#print axioms Heracles.Cost.prop_gen
#print axioms Heracles.Cost.bernstein_scaling
#print axioms Heracles.Cost.Kr_eq
#print axioms Heracles.Cost.lem_bernstein
#print axioms Heracles.Cost.lem_cost
#print axioms Heracles.Cost.lem_cost_eq
#print axioms Heracles.Cost.lem_cost_eq_converse
-- product neurons, lem:sign, prop:block
#print axioms Heracles.beta_prodNeuron
#print axioms Heracles.betaT_prodNeuron
#print axioms Heracles.lem_sign
#print axioms Heracles.prod_budget
#print axioms Heracles.prod_budget_attained
#print axioms Heracles.CbProd_eq
#print axioms Heracles.block_trig
#print axioms Heracles.prop_block
#print axioms Heracles.blockNet_table
#print axioms Heracles.blockNet_interferenceFree
#print axioms Heracles.blockNet_margin
#print axioms Heracles.blockNet_normalized_margin
#print axioms Heracles.block_attains
-- thm:cyclic
#print axioms Heracles.freq_cost
#print axioms Heracles.thm_cyclic_upper
#print axioms Heracles.thm_cyclic_attain
#print axioms Heracles.thm_cyclic_value
#print axioms Heracles.thm_cyclic_structure
-- prop:shapes parenthetical (multiplier symmetry)
#print axioms Heracles.isVertex_permVec
#print axioms Heracles.mem_Poly_permVec
#print axioms Heracles.multiset_permVec
#print axioms Heracles.budget_permVec
-- ReLU budget claim of app:models is false (counterexample, p = 5, n = 2)
#print axioms Heracles.relu_budget_claim_false
#print axioms Heracles.relu_diag_claim_false
-- power neurons (app:models, lem:sign for power neurons)
#print axioms Heracles.fhat_powNeuron
#print axioms Heracles.beta_powNeuron
#print axioms Heracles.betaT_powNeuron
#print axioms Heracles.lem_sign_pow
#print axioms Heracles.lem_sign_pow_same
#print axioms Heracles.pow_budget
#print axioms Heracles.thm_universal_pow
