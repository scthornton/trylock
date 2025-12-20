# HuggingFace AEGIS → TRYLOCK Migration Plan

**Date:** December 20, 2025
**Issue:** Mixed naming (aegis-* and trylock-*)
**Goal:** All TRYLOCK repos use "trylock-*" naming

---

## 📊 Current State

### **What You Have:**

**TRYLOCK Datasets:**
- ✅ `trylock-dataset` (PRIVATE) - 2,939 training pairs - **KEEP**
- ✅ `trylock-demo-dataset` (PUBLIC) - 48 examples - **KEEP**
- ❌ `aegis-demo-dataset` (PUBLIC) - 48 examples - **DELETE**

**TRYLOCK Models:**
- ❌ `aegis-mistral-7b-dpo` - **MIGRATE** to `trylock-mistral-7b-dpo`
- ❌ `aegis-repe-vectors` - **MIGRATE** to `trylock-repe-vectors`
- ❌ `aegis-sidecar-classifier` - **MIGRATE** to `trylock-sidecar-classifier`

**Other Projects (Keep):**
- `securecode-v2` (model + dataset)
- `atlas` (dataset)
- `chronos-*` models (4 repos)
- `bert-tiny-*` models (2 repos)

---

## 🚀 Migration Steps

### **Step 1: Create New TRYLOCK Model Repos**

#### **1.1: trylock-mistral-7b-dpo**

```bash
cd /Users/scott/perfecxion/datasets/aegis/huggingface/trylock-mistral-7b-dpo

hf upload scthornton/trylock-mistral-7b-dpo . . \
  --repo-type model \
  --create
```

**Files to upload:**
- `README.md` (✅ already updated with TRYLOCK)
- `adapter_config.json`
- `adapter_model.safetensors` (160MB)
- `training_args.bin`
- `tokenizer*` files
- `chat_template.jinja`
- `LICENSE`

---

#### **1.2: trylock-repe-vectors**

```bash
cd /Users/scott/perfecxion/datasets/aegis/huggingface/trylock-repe-vectors

hf upload scthornton/trylock-repe-vectors . . \
  --repo-type model \
  --create
```

**Files to upload:**
- `README.md` (✅ already updated)
- `repe_config.json`
- `steering_vectors.safetensors` (65KB)
- `LICENSE`

---

#### **1.3: trylock-sidecar-classifier**

```bash
cd /Users/scott/perfecxion/datasets/aegis/huggingface/trylock-sidecar-classifier

hf upload scthornton/trylock-sidecar-classifier . . \
  --repo-type model \
  --create
```

**Files to upload:**
- `README.md` (✅ already updated)
- `adapter_config.json`
- `adapter_model.safetensors` (56MB)
- `training_args.bin`
- `tokenizer*` files
- `chat_template.jinja`
- `thresholds.json`
- `LICENSE`

---

### **Step 2: Add Deprecation Notice to Old Repos**

For each old `aegis-*` repo, update the README to:

```markdown
# ⚠️ DEPRECATED - Moved to trylock-*

This repository has been renamed as part of a project rebrand.

**New Location:** [scthornton/trylock-{name}](https://huggingface.co/scthornton/trylock-{name})

Please use the new repository for the latest updates.

---

# [Original README below]
...
```

**Repos to deprecate:**
1. `aegis-mistral-7b-dpo` → Point to `trylock-mistral-7b-dpo`
2. `aegis-repe-vectors` → Point to `trylock-repe-vectors`
3. `aegis-sidecar-classifier` → Point to `trylock-sidecar-classifier`

---

### **Step 3: Delete Old Demo Dataset**

**Dataset to delete:**
- `aegis-demo-dataset` (replaced by `trylock-demo-dataset`)

**How to delete on HuggingFace:**
1. Go to: https://huggingface.co/datasets/scthornton/aegis-demo-dataset/settings
2. Scroll to "Delete this dataset"
3. Type repository name to confirm
4. Click "I understand, delete this dataset"

---

## 📝 Updated README References

After migration, update all references in your local README files:

### **In GitHub repo README:**

**Old:**
```markdown
- 🤗 [aegis-mistral-7b-dpo](https://huggingface.co/scthornton/aegis-mistral-7b-dpo)
- 🤗 [aegis-demo-dataset](https://huggingface.co/datasets/scthornton/aegis-demo-dataset)
```

**New:**
```markdown
- 🤗 [trylock-mistral-7b-dpo](https://huggingface.co/scthornton/trylock-mistral-7b-dpo)
- 🤗 [trylock-demo-dataset](https://huggingface.co/datasets/scthornton/trylock-demo-dataset)
- 🤗 [trylock-dataset](https://huggingface.co/datasets/scthornton/trylock-dataset) (Private)
```

---

## ⚡ Quick Commands Summary

### **Upload All Models:**

```bash
cd /Users/scott/perfecxion/datasets/aegis

# DPO Model
cd huggingface/trylock-mistral-7b-dpo
hf upload scthornton/trylock-mistral-7b-dpo . . --repo-type model --create
cd ../..

# RepE Vectors
cd huggingface/trylock-repe-vectors
hf upload scthornton/trylock-repe-vectors . . --repo-type model --create
cd ../..

# Sidecar Classifier
cd huggingface/trylock-sidecar-classifier
hf upload scthornton/trylock-sidecar-classifier . . --repo-type model --create
cd ../..
```

---

## ✅ Post-Migration Checklist

### **HuggingFace:**
- [ ] New `trylock-mistral-7b-dpo` created and uploaded
- [ ] New `trylock-repe-vectors` created and uploaded
- [ ] New `trylock-sidecar-classifier` created and uploaded
- [ ] Old `aegis-*` repos have deprecation notices
- [ ] Old `aegis-demo-dataset` deleted

### **GitHub:**
- [ ] Repository renamed to `trylock`
- [ ] README updated with new HuggingFace URLs
- [ ] All documentation references new repos

### **arXiv Submission:**
- [ ] Abstract references `github.com/scthornton/trylock`
- [ ] Comments field has correct GitHub URL
- [ ] No references to old "aegis" naming

---

## 📊 Final State (After Migration)

### **Datasets:**
1. ✅ `trylock-dataset` (PRIVATE - 2,939 pairs)
2. ✅ `trylock-demo-dataset` (PUBLIC - 48 examples)
3. ✅ `securecode-v2` (different project)
4. ✅ `atlas` (different project)

### **Models:**
1. ✅ `trylock-mistral-7b-dpo` (NEW)
2. ✅ `trylock-repe-vectors` (NEW)
3. ✅ `trylock-sidecar-classifier` (NEW)
4. ⚠️ `aegis-mistral-7b-dpo` (DEPRECATED)
5. ⚠️ `aegis-repe-vectors` (DEPRECATED)
6. ⚠️ `aegis-sidecar-classifier` (DEPRECATED)
7. ✅ `securecode-v2` (different project)
8. ✅ `chronos-*` models (different project)
9. ✅ `bert-tiny-*` models (different project)

---

## 💡 Why Not Just Rename?

**HuggingFace doesn't support repository renaming.**

The only options are:
1. Create new repos (what we're doing)
2. Keep old names forever (confusing)

By creating new repos and deprecating old ones:
- ✅ Clean, consistent naming
- ✅ Old links still work (redirect via deprecation notice)
- ✅ Clear migration path for users

---

## ⏱️ Estimated Time

- **Creating new repos:** 5 minutes (3 repos)
- **Uploading models:** 15-20 minutes (160MB + 56MB + 65KB)
- **Adding deprecation notices:** 5 minutes
- **Deleting old dataset:** 1 minute

**Total:** ~30 minutes

---

## 🚦 Ready to Execute?

Everything is prepared. Just run the commands in order and you'll have a clean, consistent TRYLOCK brand across all platforms!

**Next:** Run the upload commands above!
