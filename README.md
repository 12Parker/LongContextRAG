# NarrativeQA RAG Evaluation Framework

A research framework for evaluating Retrieval Augmented Generation (RAG) systems against the NarrativeQA benchmark. Compare Base LLM, Standard RAG, and Hybrid RAG approaches on long-form question-answering tasks.

## 🚀 Quick Start

### 1. Setup (5 minutes)

```bash
# Clone repository
cd LongContextRAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_narrativeqa.txt

# Configure API key
cp env.template .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

### 2. Run Comparisons

```bash
# Quick test (2 questions, 1 system)
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm \
  --num-questions 2

# Compare multiple systems (10 questions)
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm,hybrid_bm25_optimized,narrativeqa_hybrid_rag_neural \
  --num-questions 10

# Full evaluation (50+ questions)
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm,hybrid_bm25_optimized,narrativeqa_hybrid_rag_neural \
  --num-questions 50
```

**Available Systems:**
- `base_llm` - Direct LLM with full context (baseline)
- `narrativeqa_rag` - Standard RAG with vector search
- `hybrid_bm25_optimized` - Optimized BM25-only RAG (fast, low cost)
- `narrativeqa_hybrid_rag_neural` - Neural retriever RAG (best quality)
- `hybrid_bm25_dense` - Hybrid BM25 + Dense retrieval

### 3. View Results

Results are automatically saved to `results/system_comparisons/` with timestamps.

```bash
# Analyze QA metrics (F1, BERTScore, METEOR, etc.)
python scripts/analysis/analyze_qa_metrics.py

# Analyze LLM judge scores (quality assessment)
python scripts/analysis/analyze_llm_judge.py \
  --input-file results/system_comparisons/system_comparison_narrativeqa_YYYYMMDD_HHMMSS.json

# Generate comparison graphs
python scripts/analysis/generate_comparison_graphs.py

# Cost-benefit analysis
python scripts/analysis/analyze_cost_benefit.py
```

**Results Location:**
- System comparisons: `results/system_comparisons/`
- LLM judge evaluations: `results/llm_judge_evaluations/`
- Graphs: `results/graphs/`
- Cost analysis: `results/cost_benefit_analysis/`

## 📊 Example Output

After running a comparison, you'll see:

```
📊 QA SYSTEM COMPARISON
============================================================
System                    Questions  F1     BERT   METEOR   Time    
--------------------------------------------------------------------
base_llm                  50          0.087   0.000  0.110    1.00s   
hybrid_bm25_optimized     50          0.234   0.156  0.189    1.64s   
narrativeqa_hybrid_rag_neural 50     0.312   0.201  0.245    2.17s   
```

## 🎯 Common Workflows

### Quick Test
```bash
# Test one system with 5 questions
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm \
  --num-questions 5
```

### Compare Optimized Systems
```bash
# Compare the three main systems
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm,hybrid_bm25_optimized,narrativeqa_hybrid_rag_neural \
  --num-questions 50
```

### Full Analysis Pipeline
```bash
# 1. Run comparison
python scripts/comparison/compare_systems_narrativeqa.py \
  --systems base_llm,hybrid_bm25_optimized,narrativeqa_hybrid_rag_neural \
  --num-questions 50

# 2. Analyze QA metrics
python scripts/analysis/analyze_qa_metrics.py

# 3. Run LLM judge evaluation
python scripts/analysis/analyze_llm_judge.py \
  --input-file results/system_comparisons/system_comparison_narrativeqa_*.json \
  --max-questions 50

# 4. Generate graphs
python scripts/analysis/generate_comparison_graphs.py

# 5. Cost analysis
python scripts/analysis/analyze_cost_benefit.py
```

## 📁 Project Structure

```
LongContextRAG/
├── baselines/              # Baseline implementations
│   ├── narrativeqa_base_llm.py
│   └── narrativeqa_rag_baseline.py
├── hybrid/                 # Hybrid RAG systems
│   ├── narrativeqa_hybrid_rag_optimized.py
│   ├── narrativeqa_hybrid_rag_neural_retriever.py
│   └── narrativeqa_hybrid_rag_improved.py
├── scripts/
│   ├── comparison/         # System comparison scripts
│   └── analysis/           # Result analysis scripts
├── results/                # All evaluation results
│   ├── system_comparisons/
│   ├── llm_judge_evaluations/
│   ├── graphs/
│   └── cost_benefit_analysis/
└── core/                   # Configuration and prompts
```

## 🔧 Configuration

Edit `.env` file for configuration:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

## 📚 Documentation

- **[NarrativeQA Evaluation Guide](docs/NARRATIVEQA_EVALUATION_GUIDE.md)** - Detailed evaluation instructions
- **[Hybrid RAG Optimization](docs/HYBRID_RAG_OPTIMIZATION.md)** - Optimization details

## 📄 License

See LICENSE file for details.
