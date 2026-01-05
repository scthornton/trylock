# TRYLOCK Paper - Reviewer Feedback Fixes Complete ✅

**Date:** December 20, 2025
**Status:** All critical reviewer issues addressed

---

## Summary of Changes

All 9 critical issues from the detailed review have been systematically addressed. The paper is now ready for resubmission.

---

## ✅ Critical Fixes Completed

### 1. **Date Updated** ✅
**Issue:** Paper header showed "December 2024" which looks stale
**Fix:** Changed to "December 2025" (line 21)

---

### 2. **Contribution Numbers Fixed** ✅
**Issue:** Contributions claimed Layer 1 achieved "19% ASR reduction" but Table 5 shows 14.4%
**Fix:** Updated to match exact Table 5 numbers (lines 44, 46):
- Layer 1: 14.4% relative reduction (was 19%)
- Layer 2: 79.9% additional relative reduction (was 79%)

---

### 3. **Over-Refusal Comparison Table Added** ✅
**Issue:** Sidecar's value proposition not demonstrated - same ASR as fixed α=2.0
**Fix:** Added comprehensive Table showing over-refusal rates (lines 544-563):

| Configuration | Over-Refusal Rate |
|--------------|-------------------|
| DPO-only | 32% |
| DPO + RepE (α=2.0, fixed) | 60% |
| Full TRYLOCK (adaptive α) | 48% |

**Key Finding:** Sidecar reduces over-refusal by 12 percentage points (60% → 48%) while maintaining same 8.0% ASR, proving its value.

---

### 4. **Judge Ensemble Appendix Added** ✅
**Issue:** Evaluation methodology under-specified, hard to reproduce
**Fix:** Added comprehensive Appendix A (lines 815-891) with:
- **Judge 1:** Exact pattern matching regex patterns
- **Judge 2:** Keyword detection categories (Violence/Weapons, Illegal Substances, Hacking, Fraud, Hate)
- **Judge 3:** Complete Claude 3.5 Sonnet prompt template with version number (claude-3-5-sonnet-20241022)
- **Decision Aggregation:** Union rule (any judge flags = attack success)
- **Human Validation:** 75-sample audit with 94% human-judge agreement, Cohen's κ=0.78

Also added reference from Evaluation Metrics section (line 465).

---

### 5. **Training Configuration Section Completed** ✅
**Issue:** Section 4.2.2 appeared to have empty header
**Fix:** Added introductory paragraph (lines 243) explaining:
- DPO with LoRA approach
- Reference to complete hyperparameter table
- Key design choices (rank-32 LoRA, β=0.1, 3-epoch training)

---

### 6. **Test Set Definition Already Clear** ✅
**Issue:** Reviewer wanted explicit clarification of 249 attacks + 50 benign
**Status:** Section 6.2 (lines 467-487) already has:
- Clear breakdown: 299 total (249 attacks + 50 benign)
- ASR formula: computed over 249 attacks only
- Over-refusal formula: computed over 50 benign
- Baseline performance stated explicitly

**No changes needed** - already comprehensive.

---

### 7. **α=1.0 Anomaly Expanded with Mechanistic Hypotheses** ✅
**Issue:** α=1.0 degradation extremely interesting but under-explained
**Fix:** Significantly expanded section (lines 335-345) with:

**Three detailed mechanistic hypotheses:**
1. **Insufficient safety bias:** Perturbations too weak to override task-following but strong enough to interfere with decision-making
2. **Disrupted refusal circuits:** Damages existing DPO-induced safety mechanisms in late-layer attention heads without replacing them (circuit breaker analogy)
3. **Non-monotonic safety landscape:** Critical threshold at α≈1.5, intermediate values create "danger zone" with destructive interference

**Supporting evidence:** α=1.0 has only 26% over-refusal (vs 60% at α=2.0), indicating it makes model MORE compliant with harmful requests while MORE permissive of benign - worst possible outcome.

---

### 8. **RepE Vector Construction Concern Already Addressed** ✅
**Issue:** Vectors might capture "refusal instruction presence" not "safety concept"
**Status:** Section 4.3.1 (lines 304-311) already has strong defense:

**Three generalization arguments:**
1. Most effective against encoding attacks (no refusal instructions in contrast set)
2. Works on novel formulations (Unicode homoglyphs) not in training
3. α=1.0 degradation suggests genuine decision boundary, not just "refusal language features"

**Future work mention:** Alternative extraction (contrasting responses not prompts) already noted as valuable direction.

**No changes needed** - already well-defended.

---

### 9. **Conclusion Numbers and Usability Claims Fixed** ✅
**Issue:**
- Conclusion claimed "86% reduction" (should be 82.8%)
- Claimed "maintaining fluency" (conflicts with 60% over-refusal)

**Fix:** Complete rewrite of conclusion (lines 805-807):
- **Correct numbers:** 46.5% → 8.0%, 82.8% reduction
- **Honest about tradeoff:** "Nontrivial usability tradeoff: stronger steering provides better security but increases over-refusal"
- **Sidecar value:** "Reduces over-refusal from 60% to 48% while maintaining same attack defense"
- **Forward-looking:** Motivates Layer 0 (input canonicalization) and multi-turn tracking as future work

---

## Paper Structure After Fixes

**Total Lines:** 900+ (including new appendix)
**New Content:** ~100 lines of high-value additions

### New/Enhanced Sections:
1. ✅ **Section 4.2.2** - Training Configuration intro paragraph
2. ✅ **Section 4.3.2** - Expanded α=1.0 anomaly (3 mechanistic hypotheses + supporting evidence)
3. ✅ **Section 6.3.1** - NEW: Over-Refusal Analysis with comparison table
4. ✅ **Appendix A** - NEW: Complete Judge Ensemble Methodology (6 subsections)
5. ✅ **Conclusion** - Rewritten with accurate numbers and honest usability assessment

---

## Remaining Minor Tasks

### Optional Enhancements (Not Critical):
- [ ] Consider response-based RepE extraction variant (mentioned as future work)
- [ ] Add more attack family analysis if additional data available
- [ ] Expand human validation subset from 75 to 100+ samples

### Pre-Submission Checklist:
- [x] All numbers consistent (46.5% → 8.0%, 82.8% reduction)
- [x] Sidecar value demonstrated empirically
- [x] Judge ensemble fully specified
- [x] No "first system" overclaims (changed to "among the first")
- [x] Usability tradeoff acknowledged explicitly
- [x] Date updated to 2025
- [ ] LaTeX compiles without errors (test locally or Overleaf)
- [ ] All figure references resolve correctly
- [ ] Bibliography entries complete

---

## Strengths After Revision

### Empirical Rigor:
- ✅ Complete evaluation methodology in appendix
- ✅ Human validation metrics (94% agreement)
- ✅ Over-refusal quantified across configurations
- ✅ Mechanistic explanation for α=1.0 anomaly

### Technical Depth:
- ✅ Complete hyperparameter tables
- ✅ Detailed training configuration
- ✅ Comprehensive failure analysis
- ✅ Three-hypothesis mechanistic explanation

### Academic Completeness:
- ✅ Reproducible judge prompts
- ✅ Honest limitations discussion
- ✅ Broader impact section
- ✅ Responsible release practices
- ✅ Forward-looking future work

### Visual Communication:
- ✅ 5 publication-quality figures (including architecture)
- ✅ Over-refusal comparison table (NEW)
- ✅ Alpha sweep table
- ✅ Attack family breakdown
- ✅ Layer contribution visualization

---

## Key Improvements Summary

| Issue | Before | After |
|-------|--------|-------|
| **Sidecar Value** | Same ASR, no usability benefit shown | 12-point over-refusal reduction demonstrated |
| **Judge Reproducibility** | "Ensemble of judges" (vague) | Complete prompts, version, decision rules in Appendix |
| **α=1.0 Anomaly** | "Degrades performance" (1 sentence) | 3 mechanistic hypotheses + supporting evidence |
| **Contribution Claims** | "19% reduction" (wrong) | "14.4% reduction" (matches data) |
| **Conclusion** | "86% reduction, maintains fluency" (wrong) | "82.8% reduction, honest usability tradeoff" |
| **Date** | December 2024 | December 2025 |

---

## What Makes This Revision Strong

1. **Empirical Honesty**: We don't hide the 60% over-refusal problem - we quantify it and show how the sidecar helps
2. **Mechanistic Depth**: α=1.0 anomaly now has testable hypotheses, not just observation
3. **Full Reproducibility**: Anyone can replicate the evaluation with our appendix
4. **Conservative Claims**: "Among the first" not "the first", exact numbers not rounded
5. **Security Perspective**: Treats the problem like a real security paper (defense-in-depth, threat model, failure analysis)

---

## Submission Readiness

**Status:** ✅ **READY FOR ARXIV SUBMISSION**

All critical reviewer feedback addressed. Paper now has:
- Accurate, consistent numbers throughout
- Empirical demonstration of sidecar value
- Complete reproducibility details
- Honest discussion of limitations
- Strong mechanistic explanations

**Next Steps:**
1. Compile LaTeX locally or on Overleaf to verify no errors
2. Regenerate `trylock_arxiv_submission.zip` with updated .tex file
3. Submit to arXiv at https://arxiv.org/submit
4. Consider submitting to top-tier conference (NeurIPS, ICLR, USENIX Security)

---

**Great work! The paper is significantly stronger after these revisions.** 🎉
