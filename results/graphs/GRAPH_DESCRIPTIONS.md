# Comparison Graphs for Paper

This document describes the generated comparison graphs suitable for inclusion in your research paper.

## Generated Graphs

All graphs are saved in `results/graphs/` with 300 DPI resolution, suitable for publication.

### 1. `performance_comparison.png`
**Description**: Three-panel comparison of system performance metrics
- **(a) Latency Comparison**: Average response time in seconds
- **(b) Token Usage Comparison**: Average context tokens per query
- **(c) Relevance Comparison**: Average F1 relevance score

**Use Case**: Main performance comparison showing speed, efficiency, and quality trade-offs

**Caption Suggestion**: 
> "Performance comparison of RAG systems: (a) average response latency, (b) context token usage, and (c) relevance scores (F1) across 50 NarrativeQA questions."

---

### 2. `quality_scores.png`
**Description**: Grouped bar chart showing LLM judge quality scores across multiple dimensions
- Overall Score
- Correctness
- Completeness
- Relevance
- Clarity

**Use Case**: Comprehensive quality assessment using LLM-as-a-judge evaluation

**Caption Suggestion**:
> "LLM judge quality scores (0-10 scale) across five dimensions for each system. Higher scores indicate better performance."

---

### 3. `chunk_size_comparison.png`
**Description**: Three-panel analysis of chunk size impact
- **(a) Latency vs Chunk Size**: How response time changes with chunk size
- **(b) Token Usage vs Chunk Size**: How context tokens scale with chunk size
- **(c) Quality vs Chunk Size**: How quality scores vary with chunk size

**Use Case**: Parameter sensitivity analysis showing optimal chunk size selection

**Caption Suggestion**:
> "Impact of chunk size on system performance: (a) latency, (b) token usage, and (c) quality scores. Chunk size 600 provides optimal quality-latency trade-off."

---

### 4. `cost_benefit_tradeoff.png`
**Description**: Scatter plot showing cost-benefit trade-offs
- X-axis: Estimated cost per 1,000 queries (USD)
- Y-axis: LLM judge quality score
- Bubble size: Average response time
- Bubble color: Response time (darker = slower)

**Use Case**: Cost-effectiveness analysis for production deployment decisions

**Caption Suggestion**:
> "Cost-benefit trade-off analysis. Bubble size and color represent response time. Systems in the upper-left quadrant offer best quality-to-cost ratio."

---

### 5. `efficiency_metrics.png`
**Description**: Three-panel efficiency comparison
- **(a) Quality per Second**: Quality score divided by response time
- **(b) Quality per Token**: Quality score divided by token usage
- **(c) Quality per Dollar**: Quality score divided by estimated cost

**Use Case**: Efficiency metrics showing which system provides best value

**Caption Suggestion**:
> "Efficiency metrics: (a) quality per second (latency efficiency), (b) quality per token (token efficiency), and (c) quality per dollar (cost efficiency)."

---

## Graph Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (can be converted to PDF/vector if needed)
- **Style**: White grid background, serif fonts (Times New Roman)
- **Color Scheme**: Colorblind-friendly palette
- **Size**: Optimized for single-column and double-column layouts

## Usage in Paper

### Recommended Placement

1. **Performance Comparison** → Results section (main comparison)
2. **Quality Scores** → Results section (detailed quality analysis)
3. **Chunk Size Comparison** → Methodology/Results (parameter analysis)
4. **Cost-Benefit Trade-off** → Discussion section (practical implications)
5. **Efficiency Metrics** → Results/Discussion (efficiency analysis)

### Figure Numbering

When including in your paper, number them sequentially:
- Figure 1: Performance Comparison
- Figure 2: Quality Scores
- Figure 3: Chunk Size Comparison
- Figure 4: Cost-Benefit Trade-off
- Figure 5: Efficiency Metrics

## Regenerating Graphs

To regenerate graphs with different data:

```bash
python scripts/analysis/generate_comparison_graphs.py \
    --comparison-file results/system_comparisons/system_comparison_narrativeqa_YYYYMMDD_HHMMSS.json \
    --llm-judge-file results/llm_judge_evaluations/llm_judge_evaluation_YYYYMMDD_HHMMSS.json \
    --output-dir results/graphs
```

## Customization

The script uses matplotlib and seaborn. To customize:
- Colors: Modify `sns.color_palette()` calls
- Fonts: Adjust `plt.rcParams['font.family']`
- Size: Change `figsize` parameters
- Style: Modify `sns.set_style()` and `plt.rcParams`

## Notes

- All graphs use consistent color schemes across systems
- Error bars can be added if you have variance data
- Additional metrics can be plotted by modifying the script
- Graphs are saved with tight bounding boxes for easy integration

