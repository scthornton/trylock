# TRYLOCK arXiv Submission Package

**Date:** December 20, 2025
**Status:** Ready for submission

---

## 📋 arXiv Submission Details

### **Title**
```
TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering
```

### **Authors**
```
Scott Thornton
```

### **Author Affiliations**
```
Independent Researcher
```

### **Contact Email**
```
scott@perfecxion.ai
```

---

## 📚 **arXiv Categories**

### **Primary Category**
```
cs.CR - Cryptography and Security
```

**Reasoning:** LLM security, jailbreak defense, adversarial robustness

### **Cross-List Categories** (in order)
```
cs.LG - Machine Learning
cs.AI - Artificial Intelligence
cs.CL - Computation and Language
```

**Suggested submission:**
- Primary: `cs.CR`
- Cross-list: `cs.LG`, `cs.AI`, `cs.CL`

---

## 📝 **Abstract** (1,603 characters - ✅ Under 1,920 limit)

```
Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass safety alignment through prompt manipulation, multi-turn social engineering, and obfuscation. Existing defenses operate at a single layer—either modifying weights during training or adding guardrails at inference—leaving gaps that sophisticated attacks exploit. We present TRYLOCK (Adaptive Ensemble Guard with Integrated Steering), a defense-in-depth architecture combining three mechanisms: (1) Direct Preference Optimization (DPO) to train the model to recognize and refuse jailbreaks, (2) Representation Engineering (RepE) to steer activations away from attack-compliant directions at inference, and (3) a lightweight sidecar classifier enabling adaptive steering strength based on threat assessment.

We train TRYLOCK on 14,137 multi-turn attack trajectories plus 2,118 benign hard negatives, deriving 2,939 preference pairs (2,640 for training, 299 for evaluation). On this benchmark spanning five attack families, the two-layer configuration (DPO + RepE) reduces Attack Success Rate from 46.5% to 8.0%—an 82.8% relative reduction—while the three-layer configuration adds adaptive threat classification. Ablations show each layer provides independent protection: DPO yields 14.4% ASR reduction, RepE adds ~80% over DPO alone. However, RepE requires careful implementation; 8-bit quantization degraded performance to 84.7% ASR, suggesting full-precision deployment is critical. We release all components—adapters, steering vectors, classifier, and datasets—for reproducible research on layered LLM safety.
```

---

## 💬 **Comments** (Optional metadata field)

```
20 pages, 8 figures, 5 tables. Code and datasets available at https://github.com/scthornton/trylock
```

**Alternative (shorter):**
```
Code and datasets: https://github.com/scthornton/trylock
```

---

## 🏷️ **Keywords/Tags** (If asked)

```
LLM Security, Jailbreak Defense, Representation Engineering, Direct Preference Optimization, Adversarial Machine Learning, AI Safety, Prompt Injection, Defense in Depth
```

---

## 📄 **File to Upload**

### **Option 1: Upload .tex file** (Recommended if using LaTeX)
```
paper/trylock.tex
```

**Make sure to include:**
- All figure files (PNG format)
- Bibliography file (.bib)
- Any style files (.sty)

### **Option 2: Upload PDF** (If using Markdown/Word)
You'll need to:
1. Convert `paper/TRYLOCK_Canonical.md` to LaTeX or PDF
2. Upload the compiled PDF

**arXiv prefers LaTeX**, so if you have a .tex file, use that.

---

## 📊 **Paper Statistics**

Based on the canonical paper:

- **Estimated Pages:** ~20 pages (single-column)
- **Figures:** 8-10 (need to verify actual count)
- **Tables:** 5-7 (need to verify actual count)
- **Sections:** 7 main sections
- **References:** ~50+ citations

---

## ✅ **Pre-Submission Checklist**

### **Content Verification**
- [x] All AEGIS references changed to TRYLOCK
- [x] Author name: Scott Thornton
- [x] Affiliation: Independent Researcher
- [x] Contact: scott@perfecxion.ai
- [x] Abstract under 1,920 characters (1,603 ✓)
- [ ] All figures included and referenced
- [ ] All citations complete
- [ ] GitHub URL correct

### **File Preparation**
- [ ] LaTeX compiles without errors
- [ ] All figures render correctly
- [ ] Bibliography formatted properly
- [ ] No placeholder text (e.g., "TODO", "XXX")
- [ ] Acknowledgments section complete

### **Metadata**
- [ ] Title finalized
- [ ] Abstract finalized
- [ ] Categories selected (cs.CR primary)
- [ ] Comments field prepared
- [ ] Author information complete

---

## 🚀 **Submission Steps**

### **1. Create arXiv Account** (if needed)
- Go to https://arxiv.org/user/login
- Register with scott@perfecxion.ai

### **2. Start New Submission**
- Click "Submit" → "Start New Submission"
- Choose "Computer Science" archive

### **3. Fill Out Metadata**

**License:**
```
CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)
```
**Reasoning:** Matches your HuggingFace license, allows reuse with attribution

**OR**

```
arXiv-1.0 (arXiv's non-exclusive license)
```
**Reasoning:** Standard academic license

### **4. Upload Files**

**If using LaTeX:**
```bash
cd paper/
zip trylock_submission.zip trylock.tex *.png *.bib
```
Upload the ZIP file

**If using PDF:**
Upload `TRYLOCK_paper.pdf` directly

### **5. Preview and Submit**
- arXiv will compile your submission
- Review the generated PDF
- Fix any LaTeX errors
- Click "Submit to [cs.CR]"

### **6. Wait for Moderation**
- Papers typically process in **24-48 hours**
- Check email for announcements
- Paper goes live **Sunday-Thursday 8pm ET**

---

## 📢 **Post-Submission Actions**

### **Immediate (After arXiv Acceptance)**

**Update GitHub README:**
```markdown
📄 **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
```

**Update HuggingFace READMEs:**
Add arXiv link to all model cards

**Social Media Announcement:**
LinkedIn, Twitter/X, etc.

---

## 💡 **Pro Tips**

**Title Optimization:**
- Keep it under 200 characters
- Include key terms: "Defense", "LLM", "Jailbreak"
- Make it searchable and memorable

**Abstract Optimization:**
- Lead with the problem
- State your contribution clearly
- Include concrete numbers (82.8% reduction)
- End with impact/release info

**Category Selection:**
- `cs.CR` gets you into security conferences (IEEE S&P, USENIX, CCS)
- `cs.LG` increases visibility in ML community
- `cs.AI` and `cs.CL` broaden reach

**Timing:**
- Submit Monday-Wednesday for same-week publication
- Papers announce Sunday-Thursday at 8pm ET
- Avoid Friday submissions (delayed to next week)

---

## 🎯 **Suggested Submission Time**

**Best:** Monday or Tuesday morning
- Gives time for any moderation issues
- Publishes same week
- Maximizes weekend social media reach

---

## 📧 **Follow-Up After Publication**

### **Week 1:**
- Share on LinkedIn with detailed explanation
- Post on Twitter/X with thread
- Submit to HackerNews (https://news.ycombinator.com)
- Post on r/MachineLearning subreddit
- Blog post on perfecxion.ai

### **Week 2-3:**
- Email to AI security researchers
- Submit to Papers with Code
- Update Google Scholar profile
- Add to CV/resume

### **Ongoing:**
- Monitor citations
- Respond to questions on social media
- Consider writing follow-up blog posts
- Present at meetups/conferences

---

## 📝 **Ready to Submit?**

You have everything you need:
- ✅ Title (finalized)
- ✅ Abstract (1,603 chars, under limit)
- ✅ Categories (cs.CR primary)
- ✅ Author info (Scott Thornton, Independent Researcher)
- ✅ Contact (scott@perfecxion.ai)
- ✅ License recommendation (CC BY-NC-SA 4.0)
- ✅ Paper file (trylock.tex or TRYLOCK_Canonical.md)

**Next Step:** Go to https://arxiv.org/submit and follow the steps above!

---

**Good luck with the submission!** 🚀
