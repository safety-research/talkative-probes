
def quick_analyze(text: str, show_plot: bool = True, analyzer: LensAnalyzer = None):
    """Quick analysis function with optional visualization."""
    df = analyzer.analyze_all_tokens(text)

    # Print summary
    print(f"\n📊 Analysis of: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"Tokens: {len(df)}")
    print(f"Avg KL: {df['kl_divergence'].mean():.3f} (±{df['kl_divergence'].std():.3f})")
    print(f"Range: [{df['kl_divergence'].min():.3f}, {df['kl_divergence'].max():.3f}]")

    # Show tokens with explanations
    print("\nToken-by-token breakdown:")
    for _, row in df.iterrows():
        kl = row["kl_divergence"]
        mse = row["mse"]
        # Color code based on KL value
        if kl < df["kl_divergence"].quantile(0.33):
            indicator = "🟢"
        elif kl < df["kl_divergence"].quantile(0.67):
            indicator = "🟡"
        else:
            indicator = "🔴"

        print(
            f"{indicator} [{row['position']:2d}] {repr(row['token']):15} → {row['explanation']:40} (KL: {kl:.3f} MSE: {mse:.3f})"
        )

    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df["position"], df["kl_divergence"], "b-", linewidth=2, marker="o")
        plt.xlabel("Position")
        plt.ylabel("KL Divergence")
        plt.yscale("log")
        plt.title(f'KL Divergence: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)

        # Annotate some points
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(
                repr(df.iloc[i]["token"]),
                (df.iloc[i]["position"], df.iloc[i]["kl_divergence"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        plt.ylim(None, max(df["kl_divergence"][1:]))
        plt.tight_layout()
        plt.show()

    if show_plot and len(df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(df["position"], df["mse"], "r-", linewidth=2, marker="o")
        plt.xlabel("Position")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.title(f'MSE: "{text[:40]}..."')
        plt.grid(True, alpha=0.3)
        for i in range(0, len(df), max(1, len(df) // 5)):
            plt.annotate(
                repr(df.iloc[i]["token"]),
                (df.iloc[i]["position"], df.iloc[i]["mse"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
        plt.ylim(None, max(df["mse"][1:]))
        plt.tight_layout()
        plt.show()

    return df


#


def analyze_ordering_effects(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ordering effects by comparing original vs switched query results.
    """
    # Separate original and switched results
    original_results = results_df[~results_df["is_switched"]].copy()
    switched_results = results_df[results_df["is_switched"]].copy()

    # Calculate position bias (preference for A vs B)
    def calculate_position_proportion_A(answers):
        """Calculate bias toward position A"""
        valid_answers = [a for a in answers if a in ["A", "B"]]
        if not valid_answers:
            return 0.0
        return sum(1 for a in valid_answers if a == "A") / len(valid_answers)

    # Aggregate results
    ordering_analysis = []

    for _, orig_row in original_results.iterrows():
        # Find corresponding switched result
        switched_row = switched_results[switched_results["original_question_id"] == orig_row["original_question_id"]]
        print(f"looking at questions {orig_row['question_id']} and {switched_row['question_id']}")

        if len(switched_row) == 0:
            continue

        switched_row = switched_row.iloc[0]

        # Calculate metrics
        orig_neutral_a_proportion_A = calculate_position_proportion_A(orig_row["neutral_answers"])
        orig_harmful_a_proportion_A = calculate_position_proportion_A(orig_row["harmful_answers"])
        switch_neutral_a_proportion_A = calculate_position_proportion_A(switched_row["neutral_answers"])
        switch_harmful_a_proportion_A = calculate_position_proportion_A(switched_row["harmful_answers"])
        print(
            f"orig_neutral_a_proportion_A: {orig_neutral_a_proportion_A}, orig_harmful_a_proportion_A: {orig_harmful_a_proportion_A}, switch_neutral_a_proportion_A: {switch_neutral_a_proportion_A}, switch_harmful_a_proportion_A: {switch_harmful_a_proportion_A}"
        )

        # Content-based accuracy (de-biased by averaging over original and switched prompts)
        content_neutral_acc = (orig_row["neutral_accuracy"] + switched_row["neutral_accuracy"]) / 2
        content_harmful_acc = (orig_row["harmful_accuracy"] + switched_row["harmful_accuracy"]) / 2

        ordering_analysis.append(
            {
                "question_id": orig_row["original_question_id"],
                "category": orig_row["category"],
                "orig_neutral_a_proportion_A": orig_neutral_a_proportion_A,
                "orig_harmful_a_proportion_A": orig_harmful_a_proportion_A,
                "switch_neutral_a_proportion_A": switch_neutral_a_proportion_A,
                "switch_harmful_a_proportion_A": switch_harmful_a_proportion_A,
                "avg_neutral_a_bias": (orig_neutral_a_proportion_A + switch_neutral_a_proportion_A) / 2,
                "avg_harmful_a_bias": (orig_harmful_a_proportion_A + switch_harmful_a_proportion_A) / 2,
                "orig_sandbagging": orig_row["sandbagging_rate"],
                "switch_sandbagging": switched_row["sandbagging_rate"],
                "content_neutral_accuracy": content_neutral_acc,
                "content_harmful_accuracy": content_harmful_acc,
                "position_effect_neutral": abs(orig_neutral_a_proportion_A - (1 - switch_neutral_a_proportion_A)),
                "position_effect_harmful": abs(orig_harmful_a_proportion_A - (1 - switch_harmful_a_proportion_A)),
            }
        )

    return pd.DataFrame(ordering_analysis)


def plot_ordering_effects(ordering_df: pd.DataFrame):
    """
    Visualize ordering effects analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall position bias
    ax = axes[0, 0]
    contexts = ["Neutral", "Harmful"]
    avg_biases = [ordering_df["avg_neutral_a_bias"].mean(), ordering_df["avg_harmful_a_bias"].mean()]
    ax.bar(contexts, avg_biases, alpha=0.7)
    ax.axhline(y=0.5, color="r", linestyle="--", label="No bias")
    ax.set_ylabel("Average Bias Toward (A)")
    ax.set_title("Overall Position Bias by Context")
    ax.set_ylim(0, 1)
    ax.legend()

    # 2. Position effect magnitude
    ax = axes[0, 1]
    position_effects = pd.DataFrame(
        {"Neutral": ordering_df["position_effect_neutral"], "Harmful": ordering_df["position_effect_harmful"]}
    )
    position_effects.boxplot(ax=ax)
    ax.set_ylabel("Position Effect (|Original - Switched|)")
    ax.set_title("Distribution of Position Effects")

    # 3. Sandbagging comparison
    ax = axes[1, 0]
    ax.scatter(ordering_df["orig_sandbagging"], ordering_df["switch_sandbagging"], alpha=0.6)
    ax.plot([0, 1], [0, 1], "r--", label="y=x")
    ax.set_xlabel("Original Sandbagging Rate")
    ax.set_ylabel("Switched Sandbagging Rate")
    ax.set_title("Sandbagging: Original vs Switched")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    # 4. Category-wise position bias
    ax = axes[1, 1]
    category_bias = ordering_df.groupby("category")[["avg_neutral_a_bias", "avg_harmful_a_bias"]].mean()
    category_bias.plot(kind="bar", ax=ax, alpha=0.7)
    ax.axhline(y=0.5, color="r", linestyle="--")
    ax.set_ylabel("Average Bias Toward (A)")
    ax.set_title("Position Bias by Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(["Neutral", "Harmful"])

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nOrdering Effects Summary:")
    print("=" * 60)
    print(f"Average position bias (neutral): {ordering_df['avg_neutral_a_bias'].mean():.3f}")
    print(f"Average position bias (harmful): {ordering_df['avg_harmful_a_bias'].mean():.3f}")
    print(f"Average position effect (neutral): {ordering_df['position_effect_neutral'].mean():.3f}")
    print(f"Average position effect (harmful): {ordering_df['position_effect_harmful'].mean():.3f}")
    print("\nQueries with strong position effects (>0.3):")
    strong_effects = ordering_df[
        (ordering_df["position_effect_neutral"] > 0.3) | (ordering_df["position_effect_harmful"] > 0.3)
    ]
    print(f"Count: {len(strong_effects)} / {len(ordering_df)}")
