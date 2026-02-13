Key Concepts to Understand
Cosine Similarity
- Measures angle between vectors
- Range: -1 to 1 (we exponentiate to get positive values)
- Doesn't depend on magnitude, only direction

Word Embeddings (GloVe)
- Each word → fixed-length vector
- Similar words → similar vectors
- Averaging loses word order!

Sentence Embeddings (Transformers)
- Whole sentence → single vector
- Captures context and word order
- Better for semantic similarity
 
OpenAI Embeddings
- State-of-the-art quality
- Large models = better performance
- Costs money (but has free tier)

Quick Start (3 steps)
1. **Install dependencies:**
   ```bash
   pip install streamlit numpy sentence-transformers openai gdown matplotlib pandas
   ```
2. **Set OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
3. **Run the app:**
   ```bash
   streamlit run YOUR_FILE_NAME.py
   ```

# What You Need to Implement
All three tasks are marked with `# TODO` comments in the code:

Task I: `cosine_similarity(x, y)`
Task II: `averaged_glove_embeddings_gdrive()`
Task III: `get_sorted_cosine_similarity()`


How to Test
 Test 1: Basic (Make sure it runs)
- Categories: `Flowers Colors Cars Weather Food`
- Input: `"Roses are red, trucks are blue"`
- Check: All models produce results

Test 2: Sentiment
- Categories: `Positive Negative`
- Input 1: `"The movie was upsetting"`
- Input 2: `"This is terrible"`
- Check: Correct sentiment detection

# What to Include in Analysis

# Part A:
Answer these questions:
1. Which models got it right?
2. Why did some fail?
3. What does this reveal about word order?


# Part B: Model Comparison
Compare across:
- Accuracy on test cases
- Dimensionality vs performance
- Speed/response time
- Word order sensitivity


# Part C: Real-World Applications
Part C: Real-World Applications
-Pick 3 groups of everyday categories (e.g., Kitchen, Furniture, Baby, Outdoor).
-For each group, write 2 sentences using similar words but different word order.

Example:
-baby furniture safety covers → Category: Furniture
-furniture for baby room → Category: Baby / Furniture
-Observe how word position changes meaning and category assignment.
-Briefly explain why the category changes for each pair.

 Bonus: Deployment Checklist
- [ ] Push code to GitHub
- [ ] Sign up for Streamlit Cloud
- [ ] Connect GitHub repo
- [ ] Add OPENAI_API_KEY to Secrets
- [ ] Deploy!
- [ ] Test the live URL
- [ ] Submit URL in deployment.txt


Submission Checklist
Before you submit:
-  All three tasks completed and working
- Tested with at least 5 different inputs
- Analysis document (2-3 pages) written
- Comparison tables included
- Screenshots taken
- Code is commented
- README read completely
- (Bonus) App deployed and URL tested

Helpful Links
- **Streamlit Docs:** https://docs.streamlit.io
- **OpenAI API:** https://platform.openai.com
- **Sentence Transformers:** https://sbert.net
- **NumPy Docs:** https://numpy.org/doc
*Remember: The goal is to learn, not just to complete. Take your time and understand each concept.*