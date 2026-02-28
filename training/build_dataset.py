"""
Dataset Builder: Scrapes GitHub PRs to create code review training pairs

Usage:
    pip install PyGithub requests tqdm
    export GITHUB_TOKEN=your_token_here
    python build_dataset.py --output data/code_review_dataset.jsonl --limit 50000

Sources used:
    1. GitHub PR review comments (primary)
    2. Microsoft CodeReviewer dataset (fallback/supplement)
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from github import Github, GithubException

# Popular repos with high-quality code reviews
TARGET_REPOS = [
    # Python
    "psf/requests", "pallets/flask", "django/django", "tiangolo/fastapi",
    "encode/httpx", "sqlalchemy/sqlalchemy", "pydantic/pydantic",
    # JavaScript / TypeScript
    "vercel/next.js", "facebook/react", "vuejs/vue", "microsoft/TypeScript",
    # Java
    "spring-projects/spring-boot", "apache/kafka",
    # Go
    "gin-gonic/gin", "kubernetes/kubernetes",
    # Rust
    "rust-lang/rust", "tokio-rs/tokio",
]

LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
    ".c": "c", ".cs": "csharp", ".rb": "ruby", ".php": "php",
}

MIN_REVIEW_LENGTH = 50    # chars — skip trivial reviews
MIN_CODE_LINES = 3
MAX_CODE_LINES = 150      # avoid huge diffs
MAX_SAMPLES_PER_REPO = 500


def get_language(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return LANGUAGE_MAP.get(ext, "")


def extract_diff_snippet(patch: str, max_lines: int = MAX_CODE_LINES) -> str:
    """Extract added/changed lines from a git diff patch."""
    if not patch:
        return ""
    lines = patch.split("\n")
    # Get added lines (strip the leading '+')
    added = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++")]
    if len(added) < MIN_CODE_LINES or len(added) > max_lines:
        return ""
    return "\n".join(added)


def format_review_comment(comments: list) -> str:
    """Combine multiple review comments into a structured review."""
    parts = []
    for c in comments:
        body = c.body.strip()
        if len(body) >= MIN_REVIEW_LENGTH:
            parts.append(body)
    return "\n\n---\n\n".join(parts)


def scrape_github_reviews(token: str, limit: int) -> Generator[dict, None, None]:
    g = Github(token, per_page=100)
    count = 0

    for repo_name in TARGET_REPOS:
        if count >= limit:
            break
        try:
            repo = g.get_repo(repo_name)
            print(f"\n📦 Processing {repo_name}...")
            pulls = repo.get_pulls(state="closed", sort="updated", direction="desc")

            repo_count = 0
            for pr in pulls:
                if count >= limit or repo_count >= MAX_SAMPLES_PER_REPO:
                    break
                try:
                    review_comments = list(pr.get_review_comments())
                    if not review_comments:
                        continue

                    for comment in review_comments:
                        if count >= limit:
                            break

                        # Get the file context
                        filename = comment.path
                        language = get_language(filename)
                        if not language:
                            continue

                        # Get the diff snippet this comment is about
                        try:
                            files = {f.filename: f for f in pr.get_files()}
                            file_obj = files.get(filename)
                            if not file_obj:
                                continue
                            code = extract_diff_snippet(file_obj.patch or "")
                            if not code:
                                continue
                        except Exception:
                            continue

                        review_text = comment.body.strip()
                        if len(review_text) < MIN_REVIEW_LENGTH:
                            continue

                        # Build severity prefix based on keywords
                        severity = "💡 Suggestion"
                        lower = review_text.lower()
                        if any(w in lower for w in ["bug", "error", "crash", "injection", "vulnerability", "security", "unsafe", "critical"]):
                            severity = "🚨 Critical"
                        elif any(w in lower for w in ["warning", "issue", "problem", "incorrect", "wrong", "fix", "should"]):
                            severity = "⚠️ Warning"

                        # Format the output as a structured review
                        formatted_review = f"{severity} — {review_text}"

                        yield {
                            "language": language,
                            "input": code,
                            "output": formatted_review,
                            "source": f"github/{repo_name}/pr/{pr.number}",
                        }
                        count += 1
                        repo_count += 1

                    time.sleep(0.1)  # Rate limiting

                except GithubException as e:
                    if e.status == 403:
                        print("⏳ Rate limited, sleeping 60s...")
                        time.sleep(60)
                    continue

        except GithubException as e:
            print(f"❌ Error with {repo_name}: {e}")
            continue

    print(f"\n✅ Total samples collected: {count}")


def add_microsoft_codereview_data(output_path: str):
    """
    Optionally supplement with Microsoft's CodeReviewer dataset.
    Download from: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
    Format: each line is {"nl": "review comment", "code": "code diff"}
    """
    ms_path = Path("data/microsoft_codereview.jsonl")
    if not ms_path.exists():
        print("ℹ️  Microsoft CodeReviewer dataset not found, skipping.")
        return 0

    count = 0
    with open(output_path, "a") as out_f, open(ms_path) as in_f:
        for line in in_f:
            try:
                sample = json.loads(line)
                code = sample.get("code", "").strip()
                review = sample.get("nl", "").strip()
                if code and review and len(review) >= 30:
                    out_f.write(json.dumps({
                        "language": "java",  # MS dataset is primarily Java
                        "input": code,
                        "output": f"💡 Suggestion — {review}",
                        "source": "microsoft/codereview",
                    }) + "\n")
                    count += 1
            except json.JSONDecodeError:
                continue
    print(f"📚 Added {count} Microsoft CodeReviewer samples")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build code review dataset")
    parser.add_argument("--output", default="data/code_review_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    if not args.token:
        print("❌ GITHUB_TOKEN not set. Export it: export GITHUB_TOKEN=your_token")
        return

    Path("data").mkdir(exist_ok=True)
    print(f"🎯 Target: {args.limit} samples → {args.output}")

    with open(args.output, "w") as f:
        for sample in tqdm(scrape_github_reviews(args.token, args.limit), total=args.limit, desc="Scraping"):
            f.write(json.dumps(sample) + "\n")

    # Supplement with Microsoft dataset if available
    add_microsoft_codereview_data(args.output)

    # Final stats
    with open(args.output) as f:
        total = sum(1 for _ in f)
    print(f"\n🎉 Dataset complete: {total} total samples saved to {args.output}")


if __name__ == "__main__":
    main()
