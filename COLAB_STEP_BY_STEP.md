# Google Colab Evaluation - Step-by-Step Guide

## Step 1: Upload the Notebook

1. **Open your browser** and go to: https://colab.research.google.com/

2. **You should see** a welcome dialog. Click **"Upload"** tab (or if no dialog appears, click File → Upload notebook)

3. **Click "Choose File"** button

4. **Navigate to** this directory and select: `TRYLOCK_Evaluation_Colab.ipynb`

5. **Wait** for upload to complete (~5 seconds)

6. **You should see** the notebook open with multiple code cells

**✓ Success:** You'll see the notebook title "TRYLOCK Full System Evaluation (GPU)" at the top

---

## Step 2: Enable GPU

1. **Click** the menu: **Runtime → Change runtime type**

2. **A dialog will appear** titled "Notebook settings"

3. **Under "Hardware accelerator"** dropdown, select **"GPU"**
   - You might see options like "T4 GPU" or just "GPU"
   - Any GPU is fine (T4, V100, etc.)

4. **Click "Save"** button

5. **You should see** a green checkmark or "GPU" indicator in the top right

**✓ Success:** Top right corner shows "GPU" or a GPU icon

**❌ If you see "Connect" button:** Click it first, then change to GPU

---

## Step 3: Run the Evaluation

1. **Click** the menu: **Runtime → Run all**

2. **You'll see** cells start executing one by one with spinners

3. **First cell** will install dependencies (~1-2 minutes)
   - You'll see: `Installing collected packages...`
   - Wait for ✓ checkmark to appear

4. **Second cell** will prompt for file upload
   - A **"Choose Files" button** will appear
   - **STOP HERE** - Don't click it yet!

---

## Step 4: Upload Test Data

1. **When you see "Choose Files" button**, click it

2. **Navigate to** this directory: `data/dpo/test.jsonl`

3. **Select the file** and click Open

4. **You should see**: "test.jsonl (1.4 MB)" uploading
   - Progress bar will show upload (~5-10 seconds)

5. **After upload completes**, you'll see: "Loaded 299 test samples"

**✓ Success:** Message says "Loaded 299 test samples"

**❌ If error:** Make sure you uploaded `test.jsonl` from `data/dpo/` directory

---

## Step 5: Wait for Model Loading

1. **Next, you'll see**: "Loading models (this will take 5-10 minutes)..."

2. **Three loading steps:**
   ```
   [1/3] Loading DPO model...
   [2/3] Loading RepE vectors...
   [3/3] Loading sidecar classifier...
   ```

3. **Each step** downloads models from HuggingFace
   - DPO model: ~14GB (largest, takes longest)
   - RepE vectors: ~50MB (fast)
   - Sidecar: ~6GB (medium)

4. **Total time:** 5-10 minutes depending on connection

5. **You'll know it's done** when you see: "ALL MODELS READY"

**✓ Success:** All three models show ✓ checkmarks

**❌ If "CUDA out of memory" error:**
- Click Runtime → Disconnect and delete runtime
- Then Runtime → Run all again
- This clears GPU memory

---

## Step 6: Watch the Evaluation Progress

1. **You'll see**: "Evaluating 299 samples (CPU mode - will be slow)..."
   - Don't worry about "CPU mode" text - it's actually using GPU!

2. **Progress bar appears:**
   ```
   Evaluating: 45% |████████ | 135/299 [07:23<08:58, 3.28s/it]
   ```

3. **What the progress bar shows:**
   - `45%` = percentage complete
   - `135/299` = samples processed out of total
   - `07:23<08:58` = 7 min elapsed, 8 min remaining
   - `3.28s/it` = seconds per sample

4. **Typical speed:** ~3-5 seconds per sample with GPU
   - 299 samples × 4 seconds = ~20 minutes total

5. **You can minimize the tab** and check back periodically

**✓ Success:** Progress bar is moving, ETA is reasonable (~20 minutes)

**❌ If very slow (>30s per sample):** GPU may not be enabled. Check top right corner.

---

## Step 7: Get Your Results

1. **When complete**, you'll see:
   ```
   ============================================================
   TRYLOCK FULL SYSTEM EVALUATION RESULTS
   ============================================================

   Samples: 299 (XXX attacks, YYY benign)

   ASR (Attack Success Rate): XX.X%
   ORR (Over-Refusal Rate): YY.Y%

   ============================================================
   ```

2. **Copy these two numbers:**
   - ASR: _____%
   - ORR: _____%

3. **The notebook will also download** `eval_full_trylock.json`
   - Look in your Downloads folder
   - This contains detailed results for each sample

**✓ Success:** You have two percentage numbers

---

## Step 8: Share Results with Me

Just tell me:
```
ASR: XX.X%
ORR: YY.Y%
```

And I'll:
1. Add the final row to Table 2
2. Update the abstract if needed
3. Begin polishing the paper

---

## Troubleshooting

### "Runtime disconnected"
- **Fix:** Click "Reconnect" button
- This is normal - Colab times out after inactivity
- Just click Runtime → Run all again

### "GPU quota exceeded"
- **Fix:** Try again in a few hours, or use different Google account
- Free Colab has usage limits

### "File not found: test.jsonl"
- **Fix:** Make sure you uploaded the file when prompted
- Cell should show "test.jsonl (1.4 MB)" after upload

### Very slow progress (>30s per sample)
- **Fix:** Check GPU is enabled (top right corner)
- If it says "CPU", change to GPU via Runtime → Change runtime type

### "CUDA out of memory"
- **Fix:** Runtime → Disconnect and delete runtime
- Then Runtime → Run all again
- This starts fresh with clean GPU memory

### Model download stuck
- **Fix:** Click Runtime → Interrupt execution
- Then Runtime → Run all
- HuggingFace servers might be slow, retry helps

---

## What to Expect

**Timeline:**
- Step 1-2 (Upload & GPU): 2 minutes
- Step 3-5 (Start & upload data): 1 minute
- Step 6 (Model loading): 5-10 minutes
- Step 7 (Evaluation): 15-20 minutes
- **Total: 25-35 minutes**

**Normal behavior:**
- First run takes longer (downloads models)
- Progress bar may pause briefly between samples
- GPU usage will show 80-100% in Colab (good!)

**You'll know it's working when:**
- Progress bar is moving
- ETA is ~20 minutes or less
- GPU indicator shows green in top right

---

## Ready to Start?

1. Open https://colab.research.google.com/
2. Upload `TRYLOCK_Evaluation_Colab.ipynb`
3. Follow the steps above

**Let me know:**
- What you see on screen
- Any error messages
- When you get stuck

I'm here to help every step of the way!
