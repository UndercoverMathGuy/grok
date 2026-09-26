# ICLR 2027 format audit — 25 September 2026

## Result

The rebuilt manuscript passes the formatting checks below against the template
in `/Users/ruhaanrajadhyaksha/Downloads/iclr2027/`. It is **not yet a complete
submission**: the required AI-use statement is deferred at the author's request,
and Experiments and Discussion remain unfinished.

The initial-submission limit is 9 pages of main text. References and appendices
are excluded; appendices belong after the bibliography. The AI-use statement is
mandatory and also excluded from the limit. Sources:
[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
and [AI Policy for Authors](https://www.iclr.cc/Conferences/2027/AIPolicyForAuthors).
Use the explicit initial-submission rule rather than inconsistent 10-page wording
in the website's later FAQ; the supplied template also specifies 9 pages initially.

## Verified

| Check | Result |
| --- | --- |
| Official conference style and bibliography style | Byte-for-byte identical to supplied template |
| Bundled `natbib.sty`, `fancyhdr.sty`, optional math commands | Byte-for-byte identical |
| Page dimensions | Every page is 612 × 792 points, US Letter |
| Main-text length | Ends on page 7; within the 9-page limit |
| Bibliography and appendix | References on pages 7–8; appendix begins on page 9 |
| Entire PDF | 15 pages |
| Text width and margins | Official 5.5-inch text width retained; measured body text lies between x=108 and x≈504 points |
| Typography | Official Times-family body text at 10 points; official heading and abstract styles retained |
| Embedded fonts | All PDF fonts embedded; no Type 3 fonts |
| Submission mode | `\iclrfinalcopy` inactive; anonymous author block and review header |
| Review aids | Template line numbers and page numbers present |
| PDF metadata | No author name, affiliation, or local filesystem path in document metadata |
| Abstract | One paragraph in the template's abstract environment |
| Table | Centered, caption above, numbered; restored normal body font size |
| Citations | Author–year citations and official bibliography style; references alphabetically ordered |
| Compilation | Successful cached Tectonic/BibTeX build; no unresolved citations/references or overfull boxes |
| Visual review | All 15 rendered pages inspected for clipping, overlap, and layout defects |

Minor underfull-box warnings remain. These are spacing warnings, not clipped
content or alterations to the template geometry.

## Changes made

- Removed the table's explicit `\small` font reduction.
- Restored normal list spacing in the theory and appendix instead of custom
  compressed vertical spacing.
- Removed the unused `subcaption` package, preserving the conference's caption
  formatting. Retained `enumitem` because the introduction uses a label option.
- Added a source-only reminder for the author-supplied AI-use statement before
  the references. No AI-use disclosure was drafted, as requested.
- Rebuilt `paper.pdf` and its corresponding auxiliary files. The official style
  files were not modified.

## Still needed before submission

1. Supply the AI-use statement and the corresponding submission-form disclosure.
2. Complete Experiments and Discussion. The reproducibility statement currently
   says experimental settings are in the Experiments section and that code and
   configurations are supplied; those claims must match the completed materials.
3. Resolve the content scope: the abstract and finite-groups section still make
   group-theory claims, despite the proposed modular-addition focus. This audit
   did not change scientific claims or remove sections.
4. Review and anonymize the actual supplementary files. None were packaged or
   uploaded in this audit.
5. Check OpenReview-specific obligations (author information, required form
   responses, declarations, and applicable reviewing requirements). Local PDF
   inspection cannot establish those facts.

Recheck the page count and rendered layout after adding the missing material.
This audit does not constitute a fresh mathematical, empirical, bibliographic,
or Lean verification of the paper's claims.
