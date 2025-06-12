#!/usr/bin/env python3
"""
Script to process multiple prompts through RAGSystem and save results.
Reads prompts from an input file (one per line) and saves Q&A pairs to output file.
"""

import os
import sys
from datetime import datetime
from typing import List, Tuple
import argparse

from model.rag import RAGSystem


# Import your RAGSystem class
# from your_rag_module import RAGSystem  # Uncomment and adjust import path


def read_prompts(input_file: str) -> List[str]:
    """
    Read prompts from input file, one prompt per line.

    Args:
        input_file: Path to the input file containing prompts

    Returns:
        List of prompt strings
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)


def process_prompts(rag_system: 'RAGSystem', prompts: List[str]) -> List[Tuple[str, str]]:
    """
    Process all prompts through the RAG system.

    Args:
        rag_system: Instance of RAGSystem
        prompts: List of prompt strings

    Returns:
        List of (prompt, response) tuples
    """
    results = []
    total_prompts = len(prompts)

    print(f"Processing {total_prompts} prompts...")

    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{total_prompts}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        try:
            response, chunks = rag_system.ask(prompt, True)
            results.append((prompt, chunks, response))
            print(f"✓ Completed prompt {i}")
        except Exception as e:
            error_msg = f"Error processing prompt: {str(e)}"
            print(f"✗ Error on prompt {i}: {error_msg}")
            results.append((prompt, f"ERROR: {error_msg}"))

    return results


def save_results(results: List[Tuple[str, str]], output_file: str, include_metadata: bool = True):
    """
    Save the prompt-response pairs to output file.

    Args:
        results: List of (prompt, response) tuples
        output_file: Path to output file
        include_metadata: Whether to include timestamp and stats
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write("=" * 80 + "\n")
                f.write("RAG SYSTEM BATCH PROCESSING RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total prompts processed: {len(results)}\n")
                f.write("=" * 80 + "\n\n")

            for i, (prompt, chunks, response) in enumerate(results, 1):
                f.write(f"PROMPT #{i}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{prompt}\n\n")

                f.write(f"RETRIEVED CHUNKS FOR PROMPT #{i}:\n")
                f.write("-" * 40 + "\n")
                for j, chunk in enumerate(chunks, 1):
                    f.write(f"Chunk {j}:\n")
                    f.write(f"Content: {chunk.page_content}\n")
                    f.write(f"Metadata: {chunk.metadata}\n")
                    f.write("\n")
                    f.write("-" * 20 + "\n")

                f.write(f"RESPONSE #{i}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{response}\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Process prompts through RAG system and save results")
    parser.add_argument("-i", "--input", default="evaluation/input.txt",
                        help="Path to input file containing prompts (one per line)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Initialize RAG system
    print("Initializing RAG system...")
    try:
        rag_system = RAGSystem()
        rag_system.ingest("docs/Guidelines-std_complex.txt", "Guidelines-std_complex.txt")
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        sys.exit(1)

    # Read prompts
    print(f"Reading prompts from: {args.input}")
    prompts = read_prompts(args.input)
    print(f"✓ Loaded {len(prompts)} prompts")

    if not prompts:
        print("No prompts found in input file.")
        sys.exit(1)

    # Process prompts
    results = process_prompts(rag_system, prompts)

    output_file = f'evaluation/results/{args.input.split('/')[-1].split('.')[0]}_results.txt'
    # Save results
    save_results(results, output_file)

    # Print summary
    successful = sum(1 for _,_, response in results if not response.startswith("ERROR:"))
    failed = len(results) - successful

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total prompts: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()