# VERIFY (novelty_math): T5-concave-vertex, task 50

Claim (app:shape (i)): a concave function that is nondecreasing in each coordinate attains its minimum over a line-free polyhedron at a vertex. If the function is strictly concave, every minimizer is a vertex.

## Verdict: KNOWN (checker upheld)
This is a textbook fact, the concave-minimization version of Bauer's maximum principle.
- The compact case is Bauer 1958: a concave continuous function on a compact convex set attains its minimum at an extreme point. I fetched the Wikipedia page for the statement.
- The line-free, possibly unbounded polyhedral case is Rockafellar Sec. 32, "The Maximum of a Convex Function", applied to -phi. It is also standard in Horst-Tuy and in Benson's Handbook chapter.
- The paper's polyhedron P_p is unbounded, so the compact version is not enough and the no-lines version is the right one to cite. phi >= 0 is bounded below, so the polyhedral no-lines result applies even without monotonicity. The paper's self-contained proof is correct.
- Strictness step: phi is a sum of strictly concave functions of separate coordinates, so it is strictly concave on the orthant. Minimizers of a strictly concave function on a convex set are extreme points. This is folklore and needs no citation.
- No novelty is claimed and none should be. The appendix already treats it as a routine step.

## What I checked this session
- Rockafellar, Convex Analysis: Crossref DOI 10.1515/9781400873173 gives Rockafellar, PUP, 1970. The Princeton UP page gives the Princeton Mathematical Series / Landmarks reprint, and its blurb lists "the minimum or maximum of a convex function over a convex set". I could NOT fetch the Sec. 32 text: archive.org copy convexanalysis0000rock is access-restricted (inside-search 403), De Gruyter returned 405, JSTOR did not load, and Google Books API returned no hits. The section title and theorem numbers are therefore still unverified from the primary text. Cite "Sec. 32" with no theorem number and no page range.
- Horst & Tuy, Global Optimization: Crossref DOI 10.1007/978-3-662-03199-5 gives Horst, Tuy, Springer Berlin Heidelberg, 1996, ISBN 9783642082474 / 9783662031995. Metadata OK. Crossref does not list the edition, but the 1996 printing is the 3rd revised edition, so edition=3 is fine. I did not see the theorem text.
- Benson 1995 Handbook chapter: Crossref DOI 10.1007/978-1-4615-2025-2_3, pp. 43-148, confirmed. It is an optional survey citation.
- Hirsch & Hoffman 1961, "Extreme varieties, concave functions, and the fixed charge problem", CPAM 14(3):355-369, DOI 10.1002/cpa.3160140313: metadata confirmed via Crossref. This is the classical primary source for concave minimization over unbounded polyhedra at extreme points or extreme varieties. Wiley returned 403, so I did NOT verify the content and do not recommend it as confirmed.
- vom Dahl & Loehne, arXiv 1905.11228: the abstract page exists (polyhedral d.c. via concave minimization). The checker only used it as evidence of a standard technique, not as a citation.

## Issues
- None with the verdict. The recommended change (cite Rockafellar Sec. 32 and Horst-Tuy, keep the proof) is sound. Do not add a theorem number or page range (the checker's pp. 342-346 came from a search snippet and is unverified).
- Neither bib key exists yet in refs.bib (grep shows no rockafellar or horst entry), so both must be added.
- Process note: the session WebSearch budget is exhausted (200/200), so this verifier could only fetch known URLs and bibliographic APIs. Later verifiers will lose search as well.
