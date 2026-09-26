# C14 alon1994random: random-frequency lower bound as the cyclic case of Alon-Roichman

Location: sections/appendix.tex l.196 (app:scaling, "Lower bound"):
"This is the cyclic case of the random Cayley graph argument of \citet{alon1994random}."

## Bib check: CONFIRMED
- Crossref (https://api.crossref.org/works/10.1002/rsa.3240050203): "Random Cayley graphs and expanders",
  Noga Alon, Yuval Roichman, Random Structures & Algorithms 5(2):271-284, 1994.
- Semantic Scholar (DOI:10.1002/rsa.3240050203) agrees; DBLP key journals/rsa/AlonR94.
- The refs.bib entry is correct. Suggested addition: doi = {10.1002/rsa.3240050203}.

## Content check
The Crossref abstract says: for every 0<delta<1 there is a c(delta) such that, for any group G of order n and a random set S
of c(delta) log n elements, the expected second-largest eigenvalue of the normalised adjacency matrix of Cay(G,S) is at most
1-delta. So almost every such graph is an eps(delta)-expander, and the result is "essentially optimal for Abelian groups".
I could not get the full text: the Wiley page returned 403 and author-site PDFs returned 404. I therefore have no theorem
number to quote.

How it maps onto our argument:
- For G = Z_p, the eigenvalues of Cay(Z_p, {±J_l}) are the character sums (1/L) sum_l cos(2 pi J_l delta/p) for delta != 0.
- Our Hoeffding + union bound shows that all of them are <= 1/2. That is exactly the Alon-Roichman statement with 1-delta = 1/2
  and c = 8 on a cyclic group.
- For abelian groups, the Alon-Roichman proof is this same Chernoff bound over characters plus a union bound. The attribution
  is fair.
- One caveat: Alon-Roichman bound the second-LARGEST eigenvalue (one-sided). That matches our one-sided need (cos average
  <= 1/2), so the direction is right.

## Verdict: SUPPORTED
Optional improvements:
1. Their remark that the result is essentially optimal for abelian groups (Omega(log n) generators are needed) is the
   expander analogue of our matching upper bound K* = Omega(log p). It is worth one clause.
2. Per the sweep, McCracken et al. 2505.18266 Thm 4.7/Cor 4.8 (O(log n) random frequencies give an Omega(log n) logit
   margin) should be cited at the same sentence as the ML precedent.
