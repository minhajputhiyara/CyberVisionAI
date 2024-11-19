from technique_model import SecurityAnalyzer

def main():
    """Main function to run the security analyzer."""
    try:
        # Initialize analyzer
        analyzer = SecurityAnalyzer(
            model_path="model_seed_2455_best.pt",
            groq_api_key="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"
        )

        # Example text to analyze
        text = "This is an example security-related text to analyze."

        # Process and analyze
        processed_text = analyzer.preprocessor.preprocess(text)
        result = analyzer.predict(processed_text)
        groq_analysis = analyzer.get_groq_analysis(result)
        # if groq_analysis:
        #     print("\n=== Groq Security Analysis ===")
        #     print(groq_analysis)
        # else:
        #     print("No Groq analysis available.")
        
        # Return results (no print statements in main.py)
        return result,groq_analysis

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    result,groq_analysis = main()
    print(result.predicted_label)  # or handle result as needed in main.py
    print(groq_analysis)
    print(type(result.predicted_label))  # or handle result as needed in main.py
    print(type(groq_analysis))
    print(result.word_attributions)
    print(result.top_predictions)
