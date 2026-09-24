# Integration Recipe: GitHub Actions CI Automated Triage

Triage broken pull request builds in **<500 microseconds** using Axobrier on standard GitHub Actions runners (`ubuntu-latest`) in pure CPU mode (AVX2). Automatically categorize compiler errors and label PRs without burning API tokens or configuring cloud keys.

---

## 1. How It Works

1. Standard CI steps execute (`npm test`, `cargo build`, `pytest`, etc.).
2. If a build step fails, the workflow catches the terminal error output.
3. Axobrier's `build_triage.axb` cartridge runs locally on the runner's single x86_64 core using AVX2 acceleration in **~400 µs**.
4. The workflow tags the PR with an informative label (e.g., `ci:missing-dep`, `ci:syntax-error`, `ci:type-mismatch`) and posts a diagnostic comment with recommended remediation steps.

---

## 2. Complete Workflow (`.github/workflows/triage.yml`)

Copy this into your repository at `.github/workflows/triage.yml`:

```yaml
name: CI with Sub-Millisecond Axobrier Triage

on:
  pull_request:
    branches: [main, master]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write

    steps:
     - name: Checkout Code
        uses: actions/checkout@v4

     - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

     - name: Install Axobrier
        run: |
          pip install axobrier

     - name: Run Build / Test Suite
        id: build_step
        run: |
          # Run build command and stream output while capturing to log
          npm test 2>&1 | tee build_output.log
        continue-on-error: true

     - name: Sub-Millisecond Error Triage (AVX2 CPU Fallback)
        if: steps.build_step.outcome == 'failure'
        id: triage
        run: |
          # Extract the last 30 lines of error logs
          ERROR_SNIPPET=$(tail -n 30 build_output.log | tr '\n' ' ')
          
          # Evaluate against production cartridge on CPU (runs in ~400 microseconds)
          RESULT_JSON=$(axobrier route \
            --cartridge models/build_triage.axb \
            --query "$ERROR_SNIPPET")
          
          echo "triage_json=$RESULT_JSON" >> $GITHUB_OUTPUT
          
          CATEGORY=$(echo "$RESULT_JSON" | jq -r '.label')
          CONFIDENCE=$(echo "$RESULT_JSON" | jq -r '.confidence')
          LATENCY=$(echo "$RESULT_JSON" | jq -r '.latency_us')
          
          echo "category=$CATEGORY" >> $GITHUB_OUTPUT
          echo "confidence=$CONFIDENCE" >> $GITHUB_OUTPUT
          echo "latency_us=$LATENCY" >> $GITHUB_OUTPUT

     - name: Apply Diagnostic PR Label
        if: steps.build_step.outcome == 'failure'
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          CATEGORY="${{ steps.triage.outputs.category }}"
          CONFIDENCE="${{ steps.triage.outputs.confidence }}"
          
          # Convert category to label name
          LABEL_NAME="build:$CATEGORY"
          
          # Create label if it does not exist
          gh label create "$LABEL_NAME" --color "d73a4a" --description "Axobrier automated triage" --force || true
          
          # Add label to PR
          gh pr edit ${{ github.event.pull_request.number }} --add-label "$LABEL_NAME"

     - name: Post Triage Summary Comment
        if: steps.build_step.outcome == 'failure'
        uses: actions/github-script@v7
        with:
          script: |
            const category = '${{ steps.triage.outputs.category }}';
            const confidence = (parseFloat('${{ steps.triage.outputs.confidence }}') * 100).toFixed(1);
            const latency = '${{ steps.triage.outputs.latency_us }}';

            const comment = `### ⚡ Axobrier Instant Triage Diagnostic
            **Classification:** \`${category}\` (Confidence: **${confidence}%**)
            **Evaluation Latency:** \`${latency} µs\` (100% On-Runner CPU)
            **API Tokens Consumed:** \`0\` ($0.00)

            Please check compiler logs for missing modules, syntax errors, or broken assertions before re-requesting review.`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

     - name: Fail Job if Build Failed
        if: steps.build_step.outcome == 'failure'
        run: exit 1
```

---

## 3. Why Run Axobrier on CI Runners?

1. **Zero Cloud API Costs:** Traditional AI CI triage tools call OpenAI or Anthropic API endpoints on every broken build, costing $0.05 to $0.20 per workflow run. Axobrier is 100% free under Apache 2.0.
2. **Deterministic Speed:** Evaluates in **~400 µs** using standard AVX2 single-core instructions. It introduces zero measurable CI job delay.
3. **Total IP Privacy:** Sensitive proprietary codebases never send error traces or source fragments across third-party networks.
