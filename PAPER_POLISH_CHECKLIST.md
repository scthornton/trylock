# TRYLOCK Paper Polishing Checklist

## Phase 1: Post-Evaluation Updates (After Getting Numbers)

### ✅ Add Final Results
- [ ] Add Full TRYLOCK row to Table 2 with actual ASR/ORR
- [ ] Update abstract if final numbers differ from estimates
- [ ] Verify all percentage reductions are calculated correctly
- [ ] Check that all references to "full system" performance are consistent

### ✅ Numerical Consistency Check
- [ ] Verify 2,939 = 2,640 + 299 (or 2,349 + 291 + 299)
- [ ] Check all ASR/ORR percentages match across sections
- [ ] Confirm reduction percentages: (Baseline - TRYLOCK) / Baseline
- [ ] Verify Wilson score confidence intervals if reported

---

## Phase 2: Writing Quality

### Abstract
- [ ] First sentence: Clear problem statement
- [ ] Second sentence: Gap in existing solutions
- [ ] Third sentence: TRYLOCK approach
- [ ] Fourth sentence: Key results (with final numbers)
- [ ] Fifth sentence: Implications/contributions
- [ ] Length: 150-250 words
- [ ] No citations in abstract
- [ ] Self-contained (no abbreviations without definition)

### Introduction (Section 1)
- [ ] Strong opening paragraph with real-world impact
- [ ] Clear articulation of the problem
- [ ] Motivation for three-layer architecture
- [ ] Paper contributions listed explicitly
- [ ] Roadmap paragraph ("Section 2 describes...")

### Related Work (Section 2)
- [ ] Organized by category (DPO, RepE, Adaptive Defenses)
- [ ] Each work: what it does, how TRYLOCK differs
- [ ] No orphan citations (cite within sentences, not just names)
- [ ] Positioning: what's novel about TRYLOCK

### Methodology (Section 3)
- [ ] Each subsection: motivation → method → implementation
- [ ] Equations: introduced, explained, then used
- [ ] Hyperparameters: justified or cited
- [ ] Architecture diagrams referenced (if adding figures)
- [ ] Reproducibility: enough detail to reimplement

### Experiments (Section 4-5)
- [ ] Dataset: size, source, split rationale clear
- [ ] Evaluation metrics: formally defined
- [ ] Baselines: justified and fair
- [ ] Statistical significance: reported where appropriate
- [ ] Tables: clear captions, consistent formatting

### Discussion (Section 6)
- [ ] Interpret results (don't just restate numbers)
- [ ] Explain surprises (α=1.0 anomaly, sidecar performance)
- [ ] Limitations: honest and specific
- [ ] Future work: concrete and valuable
- [ ] Broader impact: considered

### Conclusion (Section 7)
- [ ] Restate problem and approach
- [ ] Highlight key result (82.8% reduction)
- [ ] Emphasize main contribution (defense-in-depth)
- [ ] Forward-looking statement
- [ ] No new information

---

## Phase 3: Technical Accuracy

### Mathematical Notation
- [ ] Variables defined before first use
- [ ] Consistent notation throughout (π_ref, not π_{ref} then pi_ref)
- [ ] Equations numbered if referenced later
- [ ] Matrix/vector notation: bold for vectors ($\mathbf{v}$), not bold for scalars ($\alpha$)
- [ ] Subscripts/superscripts: consistent style

### Terminology
- [ ] "Attack Success Rate" (ASR) - defined once, used consistently
- [ ] "Over-Refusal Rate" (ORR) - defined once, used consistently
- [ ] "Preference pairs" vs "trajectories" - clear distinction
- [ ] "Sidecar" vs "classifier" - use consistently
- [ ] "Adaptive" vs "dynamic" - pick one

### Citations
- [ ] Every claim has a citation or is demonstrated
- [ ] Citations formatted consistently
- [ ] No "et al." in first mention (unless >5 authors)
- [ ] URLs: either all in footnotes or all in references
- [ ] Self-citations: appropriate, not excessive

---

## Phase 4: Presentation

### Tables
- [ ] **Table 1:** Dataset statistics - clear and complete
- [ ] **Table 2:** Main results - final row added, best result bolded
- [ ] **Table 3:** Ablation study - interpretable
- [ ] **Table 4:** Sidecar performance - honest about limitations
- [ ] **Table 5:** Comparison to prior work - caveats clear
- [ ] All tables: caption on top, explain abbreviations
- [ ] All tables: referenced in text before appearing
- [ ] All tables: aligned columns, consistent decimal places

### Figures (if added)
- [ ] Architecture diagram: clean, professional
- [ ] Results plots: readable axis labels, legend
- [ ] Caption: describes what's shown, not what it means
- [ ] Referenced in text: "As shown in Figure X..."
- [ ] High resolution (300 DPI minimum for submission)

### Formatting
- [ ] Section headers: parallel structure (all gerunds or all nouns)
- [ ] Paragraphs: 3-6 sentences, focused on one idea
- [ ] Lists: use bullets/numbers when appropriate
- [ ] Code blocks: syntax highlighted, well-commented
- [ ] Equations: displayed (not inline) when important
- [ ] White space: balanced, not cramped

---

## Phase 5: Readability

### Sentence Structure
- [ ] Vary sentence length (not all long, not all short)
- [ ] Active voice 80%+ (prefer "we train" over "the model is trained")
- [ ] No dangling modifiers ("Using RepE, the model..." → "Using RepE, we steer...")
- [ ] Parallel construction in lists
- [ ] One main idea per sentence

### Paragraph Structure
- [ ] Topic sentence: clear main point
- [ ] Supporting sentences: evidence/explanation
- [ ] Transition sentence: link to next paragraph
- [ ] No orphan paragraphs (1 sentence)
- [ ] No monster paragraphs (>10 sentences)

### Flow Between Sections
- [ ] Each section: why this matters → what we did → what we found
- [ ] Transitions: "Having established X, we now examine Y..."
- [ ] Callbacks: reference earlier sections when relevant
- [ ] Forward references: "as we show in Section 5..."
- [ ] Logical progression: no jumps

---

## Phase 6: Polish Details

### Grammar & Style
- [ ] Spell check: technical terms in dictionary
- [ ] Hyphenation: "three-layer defense" (compound adjective)
- [ ] Capitalization: "Table 2" not "table 2", "Section 3" not "section 3"
- [ ] Numbers: spell out one-nine, numerals for 10+
- [ ] Acronyms: defined on first use in each major section
- [ ] Contractions: avoid in academic writing (it's → it is)

### Consistency
- [ ] Tense: past for experiments ("we trained"), present for facts ("TRYLOCK achieves")
- [ ] Person: consistent use of "we" (not switching to "one" or passive)
- [ ] Terminology: same term for same concept throughout
- [ ] Formatting: italics/bold used consistently
- [ ] Units: consistent (e.g., all percentages to 1 decimal place)

### Common Issues to Fix
- [ ] Remove: "clearly", "obviously", "trivially" (if clear, don't say so)
- [ ] Replace vague quantifiers: "many" → "15 out of 20"
- [ ] Avoid hedging: "seems to suggest" → "suggests" (if you have evidence)
- [ ] No future tense for contributions: "we will show" → "we show"
- [ ] Precise verbs: "utilize" → "use", "facilitate" → "enable"

---

## Phase 7: Pre-Submission Check

### Compliance
- [ ] Venue requirements: page limit, format, anonymization
- [ ] References: complete (no "Anonymous" or "Redacted")
- [ ] Supplementary material: mentioned if provided
- [ ] Ethical considerations: addressed if required
- [ ] Code/data availability: statement included

### Final Proofread
- [ ] Read entire paper aloud (catches awkward phrasing)
- [ ] Check every equation renders correctly
- [ ] Verify every citation is in references
- [ ] Click every internal reference (Section X, Table Y)
- [ ] Check page breaks (no orphan headers)
- [ ] PDF: fonts embedded, no compression artifacts

### Submission Package
- [ ] Main paper PDF
- [ ] Supplementary material (if any)
- [ ] Source files (LaTeX) if required
- [ ] README with build instructions
- [ ] Author information form
- [ ] Copyright/license agreement

---

## Polishing Priority

### Must Fix (Affects Acceptance)
- ✅ Add final evaluation numbers to Table 2
- ✅ Numerical consistency across paper
- ✅ All claims supported by evidence
- ✅ Clear contribution statement
- ✅ Honest limitations section

### Should Fix (Affects Perception)
- Clear, professional writing
- Consistent terminology
- Proper citations
- Well-formatted tables
- Smooth transitions

### Nice to Have (Affects Impact)
- Compelling abstract
- Strong introduction
- Insightful discussion
- Polished figures
- Memorable conclusion

---

## Estimated Time

- **Phase 1 (Post-evaluation):** 30 minutes
- **Phase 2 (Writing quality):** 2-3 hours
- **Phase 3 (Technical accuracy):** 1-2 hours
- **Phase 4 (Presentation):** 1 hour
- **Phase 5 (Readability):** 1-2 hours
- **Phase 6 (Polish details):** 1 hour
- **Phase 7 (Pre-submission):** 1 hour

**Total: 7-11 hours** for thorough polishing

Can be compressed to 3-4 hours if focusing on must-fix items only.

---

## Ready to Polish?

Once you have the evaluation numbers, I'll help you:
1. Add final row to Table 2
2. Work through this checklist systematically
3. Create a submission-ready version

Let me know when you have the ASR/ORR numbers!
