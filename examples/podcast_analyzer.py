#!/usr/bin/env python3
"""
Podcast and Blog Content Analyzer CLI

This script takes a podcast episode (audio/video) or blog post link,
converts it into a detailed markdown document with segmented analysis,
and exports it to Google Docs with Telegram notification.
"""

import argparse
import asyncio
import sys
from typing import Dict, Any
from datetime import datetime

from defog.llm.youtube import get_youtube_summary


class PodcastAnalyzer:
    """Main analyzer class that orchestrates the entire process"""

    def __init__(self):
        pass

    async def analyze_content(self, url: str) -> Dict[str, Any]:
        """Analyze podcast and return structured analysis"""

        # Analyze content using LLM
        print("Analyzing content segments and extracting insights...")
        analysis = await self._analyze_with_llm(url)

        return analysis

    async def _analyze_with_llm(self, url) -> str:
        """Use LLM with citations to analyze content and extract segments with insights"""

        summary = await get_youtube_summary(
            url,
            system_instructions=[
                "You are an expert content analyst specializing in technical and go-to-market (GTM) content analysis.",
                "When analyzing content:",
                "1. Focus on extracting actionable insights and hidden gems",
                "2. Pay special attention to technical details and GTM strategies",
                "3. Look for non-obvious insights that might be easily missed",
                "4. Provide exact quotes when highlighting key points",
                "5. Use proper citations to reference specific parts of the content",
                "6. Structure your analysis in a clear, comprehensive markdown format",
                "7. Always cite your sources when making claims about what was said or discussed.",
            ],
            task_description="""Analyze this podcast and provide a detailed markdown report with these sections:

# Content Analysis Report

## Executive Summary
Provide a 2-3 paragraph overview of the main themes and key takeaways.

## Detailed Segment Analysis
Break down the content into logical segments/topics, with a focus on technical and GTM content (including small/hidden details). For each segment:

### [Segment Title]
**Time/Position**: [If available from transcript or approximate position in content]
**Topic**: [Main topic discussed]
**Key Points**:
- [Detailed point 1]
- [Detailed point 2]
- [etc.]

**Technical Details** (if applicable):
- [Any technical concepts, tools, or methods discussed]
- [Code examples or technical implementations mentioned]

**GTM Insights** (if applicable):
- [Marketing strategies discussed]
- [Sales approaches mentioned]
- [Growth tactics or business insights]

## Additional Resources Mentioned
- [Links, books, tools, or other resources referenced]

Please provide detailed analysis with proper citations (ideally with timestamps) to support your insights.""",
        )

        return summary


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze podcast episodes and blog posts into detailed markdown reports"
    )
    parser.add_argument("url", help="URL of the podcast episode (YouTube) or blog post")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: podcast_analysis_[timestamp].md)",
        default=f"podcast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    )
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PodcastAnalyzer()

    try:
        # Analyze content
        print(f"Starting analysis of: {args.url}")
        analysis_data = await analyzer.analyze_content(args.url)

        # Save to file
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(analysis_data)

        print(f"Analysis saved to: {args.output}")

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
